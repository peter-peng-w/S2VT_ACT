import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/validate/test data

        # load the action
        self.action_set = {}
        with open(opt["actions_txt"], 'rt') as f:
            for line in f:
                if len(line) > 2:
                    line_data = line.strip().split(': ')
                    self.action_set[line_data[0]] = line_data[1]
        self.action_numbers = len(self.action_set)

        # load the json file which contains information about the dataset
        self.captions = json.load(open(opt["caption_json"]))
        info = json.load(open(opt["info_json"]))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of validate videos: ', len(self.splits['validate']))
        print('number of test videos: ', len(self.splits['test']))

        self.feats_dir = opt["feats_dir"]
        self.c3d_feats_dir = opt['c3d_feats_dir']
        self.with_c3d = opt['with_c3d']
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
            # Update 1 #
            instead of only return the max_len's text one-hot data, also return a action text (if there isn't, return <PAD>).
        """
        # which part of data to load
        if self.mode == 'validate':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['validate'])

        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % (ix))))
        fc_feat = np.concatenate(fc_feat, axis=1)
        if self.with_c3d == 1:
            c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video%i.npy' % (ix)))
            c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
            fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
        label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)
        captions = self.captions['video%i' % (ix)]['final_captions']
        raw_captions = self.captions['video%i' % (ix)]['captions']

        # Add <PAD>
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<EOS>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]

        # random select a caption for this video
        cap_ix = random.randint(0, len(captions) - 1)
        label = gts[cap_ix]

        # Select Actions
        cnt_actions_in_select_cap = {}     # This count the frequency of actions for the cap_ix caption
        cnt_actions_in_all_cap = {}
        select_act = '<PAD>'               # Default set to be <PAD>
        for key in self.action_set.keys():
            cnt_actions_in_select_cap[key] = 0
            cnt_actions_in_all_cap[key] = 0

        for key in self.action_set.keys():
            if self.action_set[key] in cap:
                cnt_actions_in_select_cap[key] += 1
        action_freq_select = list(cnt_actions_in_select_cap.values())

        if sum(action_freq_select) > 0:
            # randomly return an action appears in this selected caption
            act_select_idx = np.random.choice(
                self.action_numbers, 1, p=np.array(action_freq_select)/sum(action_freq_select))
            select_act = list(self.action_set.values())[act_select_idx[0]]

        else:
            for cap in raw_captions:
                for key in self.action_set.keys():
                    if self.action_set[key] in cap:
                        cnt_actions_in_all_cap[key] += 1
            action_freq_all = list(cnt_actions_in_all_cap.values())

            if sum(action_freq_all) > 0:
                act_select_idx = np.random.choice(self.action_numbers, 1, p=np.array(action_freq_all)/sum(action_freq_all))
                select_act = list(self.action_set.values())[act_select_idx[0]]

        if select_act in self.word_to_ix.keys():
            select_act_token = self.word_to_ix[select_act]
        else:
            select_act_token = self.word_to_ix['<PAD>']

        # Mask is used to mask <EOS> and <PAD>. <EOS>=1 and <PAD>=0
        non_zero = (label <= 1).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['gts'] = torch.from_numpy(gts).long()
        data['video_ids'] = 'video%i' % (ix)
        data['action'] = torch.tensor(select_act_token).type(torch.LongTensor)
        return data

    def __len__(self):
        return len(self.splits[self.mode])
