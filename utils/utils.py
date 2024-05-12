import numpy as np
import pandas as pd
import torch
import os
import sys
import shutil
from torchvision.models.resnet import ResNet, BasicBlock


def adjust_learning_rate(optimizer, epoch, init_param_lr, lr_epoch_1, lr_epoch_2):
    i = 0
    for param_group in optimizer.param_groups:
        init_lr = init_param_lr[i]
        i += 1
        if(epoch <= lr_epoch_1):
            param_group['lr'] = init_lr * 0.1 ** 0
        elif(epoch <= lr_epoch_2):
            param_group['lr'] = init_lr * 0.1 ** 1
        else:
            param_group['lr'] = init_lr * 0.1 ** 2


def load_files(directory_path):
    features = []
    labels = []
    for file_name in os.listdir(directory_path):
        if 'dataset_' in file_name:
            file = np.load(os.path.join(directory_path, file_name), allow_pickle=True)
            data = file[:, 1]
            label = file[:, 0]
            features.extend(data)
            labels.extend(label)
    temp_array = list(zip(features, labels))
    features, labels = zip(*temp_array)
    return features, labels


def load_directory_files(path, key):
    features = []
    labels = []
    if key == 'casia':
        for dir in ['casia_train', 'casia_test']:
            casia_features, casia_labels = load_files(os.path.join(path, dir))
            features.extend(list(casia_features))
            labels.extend(list(casia_labels))
        return features, labels
    elif key == 'replay':
        for dir in ['replay_mobile_train', 'replay_mobile_test']:
            replay_features, replay_labels = load_files(os.path.join(path, dir))
            features.extend(list(replay_features))
            labels.extend(list(replay_labels))
        return features, labels
    elif key == 'mobile':
        new_features, new_labels = load_files(os.path.join(path, 'mobile_new'))
        features.extend(list(new_features))
        labels.extend(list(new_labels))
        distance_features = []
        distance_labels = []
        for dir in ['spoof', 'live']:
            key_features, key_labels = load_files(os.path.join(path, 'mobile_distance', dir))
            distance_features.extend(list(key_features))
            distance_labels.extend(list(key_labels))
        features.extend(distance_features)
        labels.extend(distance_labels)
        return features, labels
    else:
        features, labels = load_files(os.path.join(path, key))
        return list(features), list(labels)


def split_fake_real(features, labels):
    df = pd.DataFrame(list(zip(features, labels)), columns=['features', 'labels'])

    df_negative = df[df['labels'] == 0].copy()
    df_positive = df[df['labels'] == 1].copy()

    features_fake = df_negative['features'].tolist()
    features_real = df_positive['features'].tolist()

    labels_fake = df_negative['labels'].tolist()
    labels_real = df_positive['labels'].tolist()

    real = [features_real, labels_real]
    fake = [features_fake, labels_fake]

    return real, fake


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdirs(checkpoint_path, best_model_path, logs):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(logs):
        os.mkdir(logs)


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def save_checkpoint(save_list, model, optimizer, gpus, model_path, filename='_checkpoint.pth.tar'):
    epoch = save_list[0]
    #valid_args = save_list[1]
    #valid_model_ACER = valid_args[0]
    #valid_model_EER = valid_args[1]
    #valid_model_HTER = round(valid_args[2], 5)
    #valid_model_AUC = valid_args[3]
    #valid_model_ACC = valid_args[4]
    #valid_model_recall = valid_args[5]
    #valid_model_precision = valid_args[6]
    #valid_model_fscore = valid_args[7]
    #valid_model_conf_matrix = valid_args[8]

    if(len(gpus) > 1):
        old_state_dict = model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            flag = k.find('.module.')
            if (flag != -1):
                k = k.replace('.module.', '.')
            new_state_dict[k] = v
        state = {
            "epoch": epoch,
            "state_dict": new_state_dict
            #"valid_arg": valid_args
            #"valid_model_ACER": valid_model_ACER,
            #"valid_model_EER": valid_model_EER,
            #"valid_model_HTER": valid_model_HTER,
            #"valid_model_AUC": valid_model_AUC,
            #"valid_model_ACC": valid_model_ACC,
            #"valid_model_recall": valid_model_recall,
            #"valid_model_precision": valid_model_precision,
            #"valid_model_fscore": valid_model_fscore,
            #"valid_model_conf_matrix": valid_model_conf_matrix
        }
    else:
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            #"valid_arg": valid_args
            #"valid_model_ACER": valid_model_ACER,
            #"valid_model_EER": valid_model_EER,
            #"valid_model_HTER": valid_model_HTER,
            #"valid_model_AUC": valid_model_AUC,
            #"valid_model_ACC": valid_model_ACC,
            #"valid_model_recall": valid_model_recall,
            #"valid_model_precision": valid_model_precision,
            #"valid_model_fscore": valid_model_fscore,
            #"valid_model_conf_matrix": valid_model_conf_matrix
        }
    #filepath = checkpoint_path + filename
    filepath = model_path + 'model' + '_' + str(epoch) + '.pth.tar'
    torch.save(state, filepath)
    #shutil.copy(filepath, model_path + 'model' + '_' + str(epoch) + '.pth.tar')


def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model_path = 'pretrained_model/resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    return model
