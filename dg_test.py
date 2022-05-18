import sys
sys.path.append('../../')
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from config import config
from utils.metrics import Metrics
from utils.dataset import SSDGDataset
from utils.utils import load_files, Logger, time_to_str
from models.DGFAS import DG_model
import time
import matplotlib.pyplot as plt

 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def test(test_dataloader, model, threshold):
    model.eval()
    prob_list = []
    label_list = []
    with torch.no_grad():
        for iter, (input, target) in enumerate(test_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            cls_out, _ = model(input, config.norm_flag)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            prob_list.extend(prob)
            label_list.extend(label)

    tn, fp, fn, tp = Metrics.confusion_matrix(label_list, prob_list, threshold)
    conf_matrix = (tn, fp, fn, tp)
    cur_acc_valid = Metrics.accuracy(conf_matrix)
    cur_precision_valid = Metrics.precision(conf_matrix)
    cur_recall_valid = Metrics.recall(conf_matrix)
    cur_fscore_valid = Metrics.f_score(cur_precision_valid, cur_recall_valid)
    cur_far_valid, cur_frr_valid = Metrics.error_rate(conf_matrix)
    cur_HTER_valid = Metrics.hter(cur_far_valid, cur_frr_valid)
    cur_ACER_valid = Metrics.acer(Metrics.apcer(label_list, prob_list), Metrics.bpcer(label_list, prob_list))
    fpr, tpr, auc_score, dist, cur_EER_valid = Metrics.roc_values(label_list, prob_list)

    Metrics.plot_histogram(label_list, prob_list, "threshold")
    plt.savefig('histogram.png')
    plt.clf()

    Metrics.plot_pr(label_list, prob_list)
    plt.savefig('pr_curve.png')
    plt.clf()

    Metrics.plot_roc(fpr, tpr, auc_score, dist, cur_EER_valid)
    plt.savefig('roc_curve.png')
    plt.clf()

    return [cur_ACER_valid, cur_EER_valid, cur_HTER_valid, auc_score,
            cur_acc_valid, cur_recall_valid, cur_precision_valid, cur_fscore_valid, (conf_matrix,)]

def main():
    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_SSDG_evaluation.txt', mode='a')
    net = DG_model(config.model).cuda()
    tgt_test_features, tgt_test_labels = load_files(os.path.join(config.data_path, config.tgt_data))
    tgt_test_data = [tgt_test_features, tgt_test_labels]
    test_dataloader = DataLoader(SSDGDataset(tgt_test_data, train=False), batch_size=1, shuffle=False)
    print('\n')
    print("**Testing** Get test files done!")
    #for file in os.listdir(config.valid_model_path):
    # load model
    #net_ = torch.load(os.path.join(config.valid_model_path + file))
    net_ = torch.load(os.path.join(config.valid_model_path + config.tgt_best_model_name))
    net.load_state_dict(net_["state_dict"])
    threshold = config.threshold
    # test model
    start = time.time()
    test_args = test(test_dataloader, net, threshold)
    end = time.time()
    print('\n===========Test Info===========\n')
    #print('model: ' + file + '\n')
    print(config.tgt_data, 'Test ACER: %5.4f' % (test_args[0] * 100))
    print(config.tgt_data, 'Test EER: %5.4f' % (test_args[1] * 100))
    print(config.tgt_data, 'Test HTER: %5.4f' % (test_args[2] * 100))
    print(config.tgt_data, 'Test AUC: %5.4f' % (test_args[3] * 100))
    print(config.tgt_data, 'Test acc: %5.4f' %(test_args[4]))
    print(config.tgt_data, 'Test recall: %5.4f' % (test_args[5]))
    print(config.tgt_data, 'Test precision: %5.4f' % (test_args[6]))
    print(config.tgt_data, 'Test fscore: %5.4f' % (test_args[7]))
    print(config.tgt_data, 'Test confusion matrix: %s' % (test_args[8]))
    print(config.tgt_data, 'Evaluation time: %s' % (time_to_str(end-start, 'sec')))
    print('\n===============================\n')

        #log.write('\n===========Test Info===========\n')
        #log.write('model: ' + file + '\n')
        #log.write('Test ACER: %5.4f\n' % (test_args[0] * 100))
        #log.write('Test EER: %5.4f\n' % (test_args[1] * 100))
        #log.write('Test HTER: %5.4f\n' % (test_args[2] * 100))
        #log.write('Test AUC: %5.4f\n' % (test_args[3] * 100))
        #log.write('Test acc: %5.4f\n' %(test_args[4]))
        #log.write('Test recall: %5.4f\n' % (test_args[5]))
        #log.write('Test precision: %5.4f\n' % (test_args[6]))
        #log.write('Test fscore: %5.4f\n' % (test_args[7]))
        #log.write('Test confusion matrix: %s\n' % (test_args[8]))
        #log.write('Evaluation time: %s\n' % (time_to_str(end-start, 'sec')))
        #log.write('\n===============================\n')

if __name__ == '__main__':
    main()
