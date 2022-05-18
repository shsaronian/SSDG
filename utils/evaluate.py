from utils.metrics import Metrics
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def eval(valid_dataloader, model, norm_flag):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    prob_list = []
    label_list = []
    #feature_list = []

    with torch.no_grad():
        for iter, (input, target) in enumerate(valid_dataloader):
            input = Variable(input).cuda()
            #input = Variable(input)
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            #target = Variable(torch.from_numpy(np.array(target)).long())
            cls_out, feature = model(input, norm_flag)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            #feature = feature.cpu().data.numpy()
            prob_list.extend(prob)
            label_list.extend(label)
            #feature_list.extend(feature)

    tn, fp, fn, tp = Metrics.confusion_matrix(label_list, prob_list, 0.5)
    conf_matrix = (tn, fp, fn, tp)
    cur_acc_valid = Metrics.accuracy(conf_matrix)
    cur_precision_valid = Metrics.precision(conf_matrix)
    cur_recall_valid = Metrics.recall(conf_matrix)
    cur_fscore_valid = Metrics.f_score(cur_precision_valid, cur_recall_valid)
    cur_far_valid, cur_frr_valid = Metrics.error_rate(conf_matrix)
    cur_HTER_valid = Metrics.hter(cur_far_valid, cur_frr_valid)
    cur_ACER_valid = Metrics.acer(Metrics.apcer(label_list, prob_list), Metrics.bpcer(label_list, prob_list))
    _, _, auc_score, _, cur_EER_valid = Metrics.roc_values(label_list, prob_list)
    #np.save('features', np.array(feature_list))

    return [cur_ACER_valid, cur_EER_valid, cur_HTER_valid, auc_score,
            cur_acc_valid, cur_recall_valid, cur_precision_valid, cur_fscore_valid, (conf_matrix,)]

