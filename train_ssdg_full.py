import sys
sys.path.append('../../')

from utils.utils import save_checkpoint, AverageMeter, Logger, mkdirs, adjust_learning_rate, time_to_str
from utils.evaluate import eval
from utils.get_loader import get_dataset
from utils.metrics import Metrics
from models.DGFAS import DG_model, Discriminator
from loss.hard_triplet_loss import HardTripletLoss
from loss.AdLoss import Real_AdLoss, Fake_AdLoss
import random
import numpy as np
from config import config
from torch.nn import functional as F
from datetime import datetime
import time
from timeit import default_timer as timer
 
import os
import torch
import torch.nn as nn
import torch.optim as optim 


random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
#os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'
#device = 'cpu'

def train():
    mkdirs(config.checkpoint_path, config.valid_model_path, config.logs)
    # load data
    src1_train_dataloader_fake, src1_train_dataloader_real, \
    src2_train_dataloader_fake, src2_train_dataloader_real, \
    src3_train_dataloader_fake, src3_train_dataloader_real, \
    src4_train_dataloader_fake, src4_train_dataloader_real, \
    src5_train_dataloader_fake, src5_train_dataloader_real, \
    src6_train_dataloader_fake, src6_train_dataloader_real, \
    tgt_valid_dataloader = get_dataset(config.src1_data,
                                       config.src2_data,
                                       config.src3_data,
                                       config.src4_data,
                                       config.src5_data,
                                       config.src6_data,
                                       config.tgt_data, config.batch_size, config.data_path)

    valid_model_ACER = 1.0
    valid_model_EER = 1.0
    valid_model_HTER = 1.0
    valid_model_AUC = 0.0
    valid_model_ACC = 0.0
    valid_model_recall = 0.0
    valid_model_precision = 0.0
    valid_model_fscore = 0.0
    valid_model_conf_matrix = 0.0

    # 0:loss, 1:ACER, 2:EER, 3:HTER, 4:AUC, 5:ACC, 6:recall 7: precision 8: fscore 9: confusion matrix
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    loss_classifier = AverageMeter()
    acc_classifier = AverageMeter()

    net = DG_model(config.model).to(device)
    ad_net_real = Discriminator().to(device)
    ad_net_fake = Discriminator().to(device)

    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_SSDG.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    print("Norm_flag: ", config.norm_flag)
    log.write('** start training target model! **\n')
    log.write(
        '--------|--- classifier ---|-------------------- validation --------------------|----------------|\n')
    log.write(
        '  iter  |    loss   ACC    | ACER  EER  HTER  AUC ACC  recall precision f-score confusion_matrix |    time      |\n')
    log.write(
        '-------------------------------------------------------------------------------------------------|\n')
    start = timer()
    criterion = {
        'softmax': nn.CrossEntropyLoss().cuda(),
        'triplet': HardTripletLoss(margin=0.1, hardest=False).cuda()
        #'softmax': nn.CrossEntropyLoss(),
        #'triplet': HardTripletLoss(margin=0.1, hardest=False)
    }
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, ad_net_real.parameters()), "lr": config.init_lr},
    ]
    optimizer = optim.SGD(optimizer_dict, lr=config.init_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    init_param_lr = []
    for param_group in optimizer.param_groups:
        init_param_lr.append(param_group["lr"])

    iter_per_epoch = 10

    src1_train_iter_real = iter(src1_train_dataloader_real)
    src1_iter_per_epoch_real = len(src1_train_iter_real)
    src2_train_iter_real = iter(src2_train_dataloader_real)
    src2_iter_per_epoch_real = len(src2_train_iter_real)
    src3_train_iter_real = iter(src3_train_dataloader_real)
    src3_iter_per_epoch_real = len(src3_train_iter_real)
    src4_train_iter_real = iter(src4_train_dataloader_real)
    src4_iter_per_epoch_real = len(src4_train_iter_real)
    src5_train_iter_real = iter(src5_train_dataloader_real)
    src5_iter_per_epoch_real = len(src5_train_iter_real)
    src6_train_iter_real = iter(src6_train_dataloader_real)
    src6_iter_per_epoch_real = len(src6_train_iter_real)

    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)
    src2_train_iter_fake = iter(src2_train_dataloader_fake)
    src2_iter_per_epoch_fake = len(src2_train_iter_fake)
    src3_train_iter_fake = iter(src3_train_dataloader_fake)
    src3_iter_per_epoch_fake = len(src3_train_iter_fake)
    src4_train_iter_fake = iter(src4_train_dataloader_fake)
    src4_iter_per_epoch_fake = len(src4_train_iter_fake)
    src5_train_iter_fake = iter(src5_train_dataloader_fake)
    src5_iter_per_epoch_fake = len(src5_train_iter_fake)
    src6_train_iter_fake = iter(src6_train_dataloader_fake)
    src6_iter_per_epoch_fake = len(src6_train_iter_fake)

    max_iter = config.max_iter
    epoch = 1
    if(len(config.gpus) > 1):
        net = torch.nn.DataParallel(net).cuda()

    for iter_num in range(max_iter+1):
        if (iter_num % src1_iter_per_epoch_real == 0):
            src1_train_iter_real = iter(src1_train_dataloader_real)
        if (iter_num % src2_iter_per_epoch_real == 0):
            src2_train_iter_real = iter(src2_train_dataloader_real)
        if (iter_num % src3_iter_per_epoch_real == 0):
            src3_train_iter_real = iter(src3_train_dataloader_real)
        if (iter_num % src4_iter_per_epoch_real == 0):
            src4_train_iter_real = iter(src4_train_dataloader_real)
        if (iter_num % src5_iter_per_epoch_real == 0):
            src5_train_iter_real = iter(src5_train_dataloader_real)
        if (iter_num % src6_iter_per_epoch_real == 0):
            src6_train_iter_real = iter(src6_train_dataloader_real)

        if (iter_num % src1_iter_per_epoch_fake == 0):
            src1_train_iter_fake = iter(src1_train_dataloader_fake)
        if (iter_num % src2_iter_per_epoch_fake == 0):
            src2_train_iter_fake = iter(src2_train_dataloader_fake)
        if (iter_num % src3_iter_per_epoch_fake == 0):
            src3_train_iter_fake = iter(src3_train_dataloader_fake)
        if (iter_num % src4_iter_per_epoch_fake == 0):
            src4_train_iter_fake = iter(src4_train_dataloader_fake)
        if (iter_num % src5_iter_per_epoch_fake == 0):
            src5_train_iter_fake = iter(src5_train_dataloader_fake)
        if (iter_num % src6_iter_per_epoch_fake == 0):
            src6_train_iter_fake = iter(src6_train_dataloader_fake)

        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
        param_lr_tmp = []
        for param_group in optimizer.param_groups:
            param_lr_tmp.append(param_group["lr"])

        net.train(True)
        ad_net_real.train(True)
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, init_param_lr, config.lr_epoch_1, config.lr_epoch_2)
        ######### data prepare #########
        src1_img_real, src1_label_real = src1_train_iter_real.next()
        src1_img_real = src1_img_real.cuda()
        src1_label_real = src1_label_real.cuda()
        input1_real_shape = src1_img_real.shape[0]

        src2_img_real, src2_label_real = src2_train_iter_real.next()
        src2_img_real = src2_img_real.cuda()
        src2_label_real = src2_label_real.cuda()
        input2_real_shape = src2_img_real.shape[0]

        src3_img_real, src3_label_real = src3_train_iter_real.next()
        src3_img_real = src3_img_real.cuda()
        src3_label_real = src3_label_real.cuda()
        input3_real_shape = src3_img_real.shape[0]

        src4_img_real, src4_label_real = src4_train_iter_real.next()
        src4_img_real = src4_img_real.cuda()
        src4_label_real = src4_label_real.cuda()
        input4_real_shape = src4_img_real.shape[0]

        src5_img_real, src5_label_real = src5_train_iter_real.next()
        src5_img_real = src5_img_real.cuda()
        src5_label_real = src5_label_real.cuda()
        input5_real_shape = src5_img_real.shape[0]

        src6_img_real, src6_label_real = src6_train_iter_real.next()
        src6_img_real = src6_img_real.cuda()
        src6_label_real = src6_label_real.cuda()
        input6_real_shape = src6_img_real.shape[0]

        src1_img_fake, src1_label_fake = src1_train_iter_fake.next()
        src1_img_fake = src1_img_fake.cuda()
        src1_label_fake = src1_label_fake.cuda()
        input1_fake_shape = src1_img_fake.shape[0]

        src2_img_fake, src2_label_fake = src2_train_iter_fake.next()
        src2_img_fake = src2_img_fake.cuda()
        src2_label_fake = src2_label_fake.cuda()
        input2_fake_shape = src2_img_fake.shape[0]

        src3_img_fake, src3_label_fake = src3_train_iter_fake.next()
        src3_img_fake = src3_img_fake.cuda()
        src3_label_fake = src3_label_fake.cuda()
        input3_fake_shape = src3_img_fake.shape[0]

        src4_img_fake, src4_label_fake = src4_train_iter_fake.next()
        src4_img_fake = src4_img_fake.cuda()
        src4_label_fake = src4_label_fake.cuda()
        input4_fake_shape = src4_img_fake.shape[0]

        src5_img_fake, src5_label_fake = src5_train_iter_fake.next()
        src5_img_fake = src5_img_fake.cuda()
        src5_label_fake = src5_label_fake.cuda()
        input5_fake_shape = src5_img_fake.shape[0]

        src6_img_fake, src6_label_fake = src6_train_iter_fake.next()
        src6_img_fake = src6_img_fake.cuda()
        src6_label_fake = src6_label_fake.cuda()
        input6_fake_shape = src6_img_fake.shape[0]

        input_data = torch.cat([src1_img_real, src1_img_fake,
                                src2_img_real, src2_img_fake,
                                src3_img_real, src3_img_fake,
                                src4_img_real, src4_img_fake,
                                src5_img_real, src5_img_fake,
                                src6_img_real, src6_img_fake], dim=0)

        source_label = torch.cat([src1_label_real, src1_label_fake,
                                  src2_label_real, src2_label_fake,
                                  src3_label_real, src3_label_fake,
                                  src4_label_real, src4_label_fake,
                                  src5_label_real, src5_label_fake,
                                  src6_label_real, src6_label_fake], dim=0)

        ######### forward #########
        classifier_label_out, feature = net(input_data, config.norm_flag)

        ######### single side adversarial learning #########
        input1_shape = input1_real_shape + input1_fake_shape
        input2_shape = input2_real_shape + input2_fake_shape
        input3_shape = input3_real_shape + input3_fake_shape
        input4_shape = input4_real_shape + input4_fake_shape
        input5_shape = input5_real_shape + input5_fake_shape
        feature_real_1 = feature.narrow(0, 0, input1_real_shape)
        feature_real_2 = feature.narrow(0, input1_shape, input2_real_shape)
        feature_real_3 = feature.narrow(0, input1_shape+input2_shape, input3_real_shape)
        feature_real_4 = feature.narrow(0, input1_shape + input2_shape + input3_shape, input4_real_shape)
        feature_real_5 = feature.narrow(0, input1_shape + input2_shape + input3_shape + input4_shape, input5_real_shape)
        feature_real_6 = feature.narrow(0, input1_shape + input2_shape + input3_shape + input4_shape + input5_shape, input6_real_shape)
        feature_real = torch.cat([feature_real_1, feature_real_2, feature_real_3,
                                  feature_real_4, feature_real_5, feature_real_6], dim=0)
        discriminator_out_real = ad_net_real(feature_real)

        ######### unbalanced triplet loss #########
        real_domain_label_1 = torch.LongTensor(input1_real_shape, 1).fill_(0).cuda()
        #real_domain_label_1 = torch.LongTensor(input1_real_shape, 1).fill_(0)
        real_domain_label_2 = torch.LongTensor(input2_real_shape, 1).fill_(0).cuda()
        #real_domain_label_2 = torch.LongTensor(input2_real_shape, 1).fill_(0)
        real_domain_label_3 = torch.LongTensor(input3_real_shape, 1).fill_(0).cuda()
        #real_domain_label_3 = torch.LongTensor(input3_real_shape, 1).fill_(0)
        real_domain_label_4 = torch.LongTensor(input4_real_shape, 1).fill_(0).cuda()
        #real_domain_label_4 = torch.LongTensor(input4_real_shape, 1).fill_(0)
        real_domain_label_5 = torch.LongTensor(input5_real_shape, 1).fill_(0).cuda()
        #real_domain_label_5 = torch.LongTensor(input5_real_shape, 1).fill_(0)
        real_domain_label_6 = torch.LongTensor(input6_real_shape, 1).fill_(0).cuda()
        #real_domain_label_6 = torch.LongTensor(input6_real_shape, 1).fill_(0)

        fake_domain_label_1 = torch.LongTensor(input1_fake_shape, 1).fill_(1).cuda()
        #fake_domain_label_1 = torch.LongTensor(input1_fake_shape, 1).fill_(1)
        fake_domain_label_2 = torch.LongTensor(input2_fake_shape, 1).fill_(2).cuda()
        #fake_domain_label_2 = torch.LongTensor(input2_fake_shape, 1).fill_(2)
        fake_domain_label_3 = torch.LongTensor(input3_fake_shape, 1).fill_(3).cuda()
        #fake_domain_label_3 = torch.LongTensor(input3_fake_shape, 1).fill_(3)
        fake_domain_label_4 = torch.LongTensor(input4_fake_shape, 1).fill_(4).cuda()
        #fake_domain_label_4 = torch.LongTensor(input4_fake_shape, 1).fill_(4)
        fake_domain_label_5 = torch.LongTensor(input5_fake_shape, 1).fill_(5).cuda()
        #fake_domain_label_5 = torch.LongTensor(input5_fake_shape, 1).fill_(5)
        fake_domain_label_6 = torch.LongTensor(input6_fake_shape, 1).fill_(6).cuda()
        #fake_domain_label_6 = torch.LongTensor(input6_fake_shape, 1).fill_(6)

        source_domain_label = torch.cat([real_domain_label_1, fake_domain_label_1,
                                         real_domain_label_2, fake_domain_label_2,
                                         real_domain_label_3, fake_domain_label_3,
                                         real_domain_label_4, fake_domain_label_4,
                                         real_domain_label_5, fake_domain_label_5,
                                         real_domain_label_6, fake_domain_label_6], dim=0).view(-1)
        triplet = criterion["triplet"](feature, source_domain_label)

        ######### cross-entropy loss #########
        real_shape_list = []
        real_shape_list.append(input1_real_shape)
        real_shape_list.append(input2_real_shape)
        real_shape_list.append(input3_real_shape)
        real_shape_list.append(input4_real_shape)
        real_shape_list.append(input5_real_shape)
        real_shape_list.append(input6_real_shape)

        real_adloss = Real_AdLoss(discriminator_out_real, criterion["softmax"], real_shape_list)
        cls_loss = criterion["softmax"](classifier_label_out.narrow(0, 0, input_data.size(0)), source_label)

        ######### backward #########
        total_loss = cls_loss + config.lambda_triplet * triplet + config.lambda_adreal * real_adloss 
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_classifier.update(cls_loss.item())
        probs = F.softmax(classifier_label_out.narrow(0, 0, input_data.size(0)), dim=1).cpu().data.numpy()[:, 1]
        tn, fp, fn, tp = Metrics.confusion_matrix(source_label.cpu().data.numpy(), probs, config.threshold)
        conf_matrix = (tn, fp, fn, tp)
        acc = Metrics.accuracy(conf_matrix)
        acc_classifier.update(acc)

        print('\r', end='', flush=True)
        print(
            '  %4.1f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %s  | %s'
            % (
                (iter_num+1) / iter_per_epoch, loss_classifier.avg, acc_classifier.avg,
                float(valid_model_ACER * 100), float(valid_model_EER * 100),
                float(valid_model_HTER * 100), float(valid_model_AUC * 100),
                float(valid_model_ACC), float(valid_model_recall),
                float(valid_model_precision), float(valid_model_fscore),
                valid_model_conf_matrix,
                time_to_str(timer() - start, 'min'))
            , end='', flush=True)

        if (iter_num != 0 and (iter_num+1) % 20 == 0):
            # 0:ACER, 1:EER, 2:HTER, 3:AUC, 4:ACC, 5:recall 6: precision 7: f-score 8: confusion matrix
            valid_args = eval(tgt_valid_dataloader, net, config.norm_flag)
            # judge model according to HTER
            #is_best = valid_args[2] <= best_model_HTER
            #best_model_HTER = min(valid_args[2], best_model_HTER)
            #if (valid_args[2] <= best_model_HTER):
            #    best_model_ACER = valid_args[0]
            #    best_model_EER = valid_args[1]
            #    best_model_AUC = valid_args[3]
            #    best_model_ACC = valid_args[4]
            #    best_model_recall = valid_args[5]
            #    best_model_precision = valid_args[6]
            #    best_model_fscore = valid_args[7]
            #    best_model_conf_matrix = valid_args[8]
            valid_model_ACER = valid_args[0]
            valid_model_EER = valid_args[1]
            valid_model_HTER = valid_args[2]
            valid_model_AUC = valid_args[3]
            valid_model_ACC = valid_args[4]
            valid_model_recall = valid_args[5]
            valid_model_precision = valid_args[6]
            valid_model_fscore = valid_args[7]
            valid_model_conf_matrix = valid_args[8]

            #save_list = [epoch, valid_args, best_model_ACER, best_model_EER, best_model_HTER, best_model_AUC,
            #             best_model_ACC, best_model_recall, best_model_precision, best_model_fscore, best_model_conf_matrix]
            save_list = [epoch, valid_args]
            save_checkpoint(save_list, net, config.gpus, config.checkpoint_path, config.valid_model_path)
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %s| %s'
                % (
                (iter_num + 1) / iter_per_epoch, loss_classifier.avg, acc_classifier.avg,
                float(valid_model_ACER * 100), float(valid_model_EER * 100),
                float(valid_model_HTER * 100), float(valid_model_AUC * 100),
                float(valid_model_ACC), float(valid_model_recall),
                float(valid_model_precision), float(valid_model_fscore),
                valid_model_conf_matrix,
                time_to_str(timer() - start, 'min')))
            log.write('\n')
            time.sleep(0.01)

if __name__ == '__main__':
    train()




















