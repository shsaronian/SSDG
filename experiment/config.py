class DefaultConfigs(object):
    seed = 666
    threshold = 0.2
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
    max_iter = 4000
    #max_iter = 10
    lambda_triplet = 2
    lambda_adreal = 0.1
    # test model name
    tgt_best_model_name = 'model_best_0.08_29.pth.tar' 
    # source data information
    src1_data = 'mobile_distance'
    src2_data = 'msumfsd'
    src3_data = 'nuaa'
    src4_data = 'pc'
    src5_data = 'replay_mobile'
    src6_data = 'rose'
    # target data information
    tgt_data = 'casia'
    tgt_test_num_frames = 1
    # paths information
    data_path = "D:/Sharon's files/SSDG-CVPR2020/data"
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = './' + tgt_data + '_checkpoint/' + model + '/best_model/'
    logs = './logs/'

config = DefaultConfigs()
