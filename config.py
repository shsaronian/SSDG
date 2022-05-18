class DefaultConfigs(object):
    seed = 666
    threshold = 0.5
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.01
    lr_epoch_1 = 0
    lr_epoch_2 = 150
    # model
    pretrained = True
    model = 'resnet18'     # resnet18 or maddg
    # training parameters
    gpus = "1"
    batch_size = 10
    norm_flag = True
    max_iter = 3000
    lambda_triplet = 2
    lambda_adreal = 0.1
    # test model name
    tgt_best_model_name = 'model_valid_98.pth.tar'
    # source data information
    src1_data = 'casia_train'
    src2_data = 'mobile_distance'
    src3_data = 'msumfsd'
    src4_data = 'nuaa'
    src5_data = 'pc'
    src6_data = 'replay_mobile_train'
    # target data information
    tgt_data = 'data_val'
    # paths information
    data_path = "./data/"
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    valid_model_path = './' + tgt_data + '_checkpoint/' + model + '/valid_model/'
    #valid_model_path = './' + 'data_val' + '_checkpoint/' + model + '/valid_model/'
    logs = './logs/'

config = DefaultConfigs()
