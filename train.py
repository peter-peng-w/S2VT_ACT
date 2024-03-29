import json
import os
from os import path
import numpy as np

import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset, VideoActDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from model.S2VTModel import S2VTModel
from model.S2VTACTModel import S2VTACTModel
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb


def train(loader, model, optimizer, lr_scheduler, opt, device, crit):
    # Add Tensorboard
    train_logger, valid_logger = None, None
    if opt["log_dir"] is not None:
        train_logger = tb.SummaryWriter(path.join(opt["log_dir"], 'train'), flush_secs=1)
        # valid_logger = tb.SummaryWriter(path.join(opt["log_dir"], 'valid'), flush_secs=1)

    if opt["model"] == 'S2VTACTModel':
        use_action = True
    else:
        use_action = False

    # Training Procedure
    model.train()
    global_step = 0
    # model = nn.DataParallel(model) # just ignore data parallel here
    for epoch in range(opt["epochs"]):
        iteration = 0
        # If start self crit training
        if opt["self_crit_after"] != -1 and epoch >= opt["self_crit_after"]:
            sc_flag = True
            init_cider_scorer(opt["cached_tokens"])
        else:
            sc_flag = False

        for data in loader:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fc_feats = data['fc_feats'].to(device)
            labels = data['labels'].to(device)
            masks = data['masks'].to(device)

            if use_action:
                action = data['action'].to(device)

            optimizer.zero_grad()
            if not sc_flag:
                if use_action:
                    seq_probs, _ = model(vid_feats=fc_feats, action=action, device=device, target_variable=labels, mode='train')
                else:
                    seq_probs, _ = model(vid_feats=fc_feats, target_variable=labels, mode='train')
                # Using Language Model Loss (NLLLoss or CrossEntropy Loss)
                loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            else:
                print('Currently ignore RL criterion')
                # seq_probs, seq_preds = model(
                #     fc_feats, mode='inference', opt=opt)
                # reward = get_self_critical_reward(model, fc_feats, data,
                #                                   seq_preds)
                # print(reward.shape)
                # loss = rl_crit(seq_probs, seq_preds,
                #                torch.from_numpy(reward).float().cuda())

            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            iteration += 1
            global_step += 1

            if not sc_flag:
                if iteration % 20 == 0:
                    print("iter %d (epoch %d), train_loss = %.6f" % (iteration, epoch, train_loss))
            else:
                print('Currently ignore RL criterion')
                # print("iter %d (epoch %d), avg_reward = %.6f" %
                #       (iteration, epoch, np.mean(reward[:, 0])))

            # Add Logger
            if train_logger is not None and global_step % 100 == 0:
                # Log some real data
                pass

            # Add Loss Statistics
            if train_logger is not None and iteration % 10 == 0:
                train_logger.add_scalar('loss', train_loss, global_step=global_step)

        # Step the Learning Rate Scheduler
        lr_scheduler.step()
        train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if epoch % opt["save_checkpoint_every"] == 0:
            model_path = os.path.join(opt["checkpoint_path"], 'model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"], 'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))


def main(opt):
    # DataLoader
    if opt["model"] == 'S2VTModel':
        dataset = VideoDataset(opt, 'train')
    elif opt["model"] == 'S2VTACTModel':
        dataset = VideoActDataset(opt, 'train')
    else:
        print('Currently Not Support this model: {}'.format(opt["model"]))
        raise ValueError
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    opt["vocab_size"] = dataset.get_vocab_size()

    if opt["model"] == 'S2VTModel':
        print(opt)
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])

    elif opt["model"] == 'S2VTACTModel':
        print(opt)
        model = S2VTACTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])

    elif opt["model"] == "S2VTAttModel":
        print('Currently not supported.')
        raise ValueError
    # Load model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    # Criterions #
    LMCriterion = utils.LanguageModelCriterion()
    # rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(dataloader, model, optimizer, exp_lr_scheduler, opt, device, LMCriterion)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)
