import re
import json
import argparse
import numpy as np


def build_vocab(vids, params):
    # Load all the actions
    action_set = {}
    with open('./data/actions_short.txt', 'rt') as f:
        for line in f:
            if len(line) > 2:
                line_data = line.strip().split(': ')
                action_set[line_data[0]] = line_data[1]
    # Threshold of the word frequency
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    for vid, caps in vids.items():
        for cap in caps['captions']:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            for w in ws:
                counts[w] = counts.get(w, 0) + 1
    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    # Add the action set into total vocabulary
    for word in action_set.values():
        counts[word] = counts.get(word, 0) + 1
    # Filter seldom appear words
    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('ratio of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    # let's now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')
    for vid, caps in vids.items():
        caps = caps['captions']
        vids[vid]['final_captions'] = []
        for cap in caps:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            caption = [
                '<SOS>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<EOS>']
            vids[vid]['final_captions'].append(caption)
    return vocab


def main(params):
    sentences = json.load(open(params['input_json'], 'r'))['sentences']
    video_caption = {}
    # Load captions for each video (key: video_id, value: [captions])
    for s in sentences:
        if s['video_id'] not in video_caption.keys():
            video_caption[s['video_id']] = {'captions': []}
        video_caption[s['video_id']]['captions'].append(s['caption'])
    # Create the vocab
    vocab = build_vocab(video_caption, params)
    # Build tokenization mapping
    itow = {i + 3: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 3 for i, w in enumerate(vocab)}  # inverse table
    # Add <SOS>, <EOS> and <PAD>
    wtoi['<EOS>'] = 1
    itow[0] = '<EOS>'
    wtoi['<SOS>'] = 2
    itow[1] = '<SOS>'
    wtoi['<PAD>'] = 0
    itow[2] = '<PAD>'

    out = {}
    out['ix_to_word'] = itow
    out['word_to_ix'] = wtoi
    out['videos'] = {'train': [], 'validate': [], 'test': []}
    videos = json.load(open(params['input_json'], 'r'))['videos']
    for i in videos:
        out['videos'][i['split']].append(int(i['id']))
    json.dump(out, open(params['info_json'], 'w'), indent=4)
    json.dump(video_caption, open(params['caption_json'], 'w'), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', type=str, default='data/MSR-VTT_Lite/data_lite.json',
                        help='msr_vtt videoinfo json')
    parser.add_argument('--info_json', default='data/info.json',
                        help='info about iw2word and word2ix')
    parser.add_argument('--caption_json', default='data/caption.json', help='caption json file')
    parser.add_argument('--word_count_threshold', default=1, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
