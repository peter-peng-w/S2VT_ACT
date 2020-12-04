import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.S2VTModel import S2VTModel
from model.S2VTACTModel import S2VTACTModel
from dataloader import VideoDataset, VideoActDataset
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer

from pandas.io.json import json_normalize


def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
    return gts


def test(model, crit, dataset, vocab, device, opt):
    model.eval()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    scorer = COCOScorer()
    gt_dataframe = json_normalize(
        json.load(open(opt["input_json"]))['sentences'])
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    results = []
    samples = {}
    for data in loader:
        # forward the model to get loss
        fc_feats = data['fc_feats'].to(device)
        labels = data['labels'].to(device)
        masks = data['masks'].to(device)
        video_ids = data['video_ids']
        if opt["model"] == "S2VTACTModel":
            action = data['action'].to(device)
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            if opt["model"] == "S2VTModel":
                seq_probs, seq_preds = model(
                    fc_feats, mode='inference', opt=opt)
            else:
                seq_probs, seq_preds = model(
                    fc_feats, action=action, device=device, mode='inference', opt=opt)

        sents = utils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    results.append(valid_score)
    print(valid_score)

    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])

    with open(os.path.join(opt["results_path"], "scores.txt"), 'a') as scores_table:
        scores_table.write(json.dumps(results[0]) + "\n")
    with open(os.path.join(opt["results_path"], opt["model"].split("/")[-1].split('.')[0] + ".json"), 'w') as prediction_results:
        json.dump({"predictions": samples, "scores": valid_score}, prediction_results)


def main(opt):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if opt["model"] == "S2VTModel":
        dataset = VideoDataset(opt, "test")
    elif opt["model"] == "S2VTACTModel":
        dataset = VideoActDataset(opt, "test")
    else:
        print('Currently not supported: {}'.format(opt["model"]))
        raise ValueError
    
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len
    if opt["model"] == "S2VTModel":
        model = S2VTModel(
            opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
            rnn_dropout_p=opt["rnn_dropout_p"]).to(device)
    elif opt["model"] == "S2VTACTModel":
        model = S2VTACTModel(
            opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
            rnn_dropout_p=opt["rnn_dropout_p"]).to(device)
    elif opt["model"] == "S2VTAttModel":
        print('Currently Not Supported: {}'.format(opt["model"]))
        raise ValueError
    # model = nn.DataParallel(model)
    # Setup the model
    model.load_state_dict(torch.load(opt["saved_model"]))
    crit = utils.LanguageModelCriterion()

    test(model, crit, dataset, dataset.get_vocab(), device, opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recover_opt', type=str, required=True,
                        help='recover train opts from saved opt_json')
    parser.add_argument('--saved_model', type=str, default='',
                        help='path to saved model to evaluate')
    parser.add_argument('--dump_json', type=int, default=1,
                        help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--dump_path', type=int, default=0,
                        help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device number')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--sample_max', type=int, default=1,
                        help='0/1. whether sample max probs  to get next word in inference stage')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1. Usually 2 or 3 works well.')

    args = parser.parse_args()
    args = vars((args))
    opt = json.load(open(args["recover_opt"]))
    for k, v in args.items():
        opt[k] = v
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    main(opt)
