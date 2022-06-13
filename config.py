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
    max_iter = 4000
    lambda_triplet = 2
    lambda_adreal = 0.1
    # source data information
    src1_data = 'pc'
    src2_data = 'mobile'
    src3_data = 'msu'
    src4_data = 'casia'
    #src5_data = 'replay_mobile'
    #src6_data = 'rose'
    # target data information
    tgt_data = 'rose'
    tgt_test_num_frames = 1
    # paths information
    data_path = "./data/"
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    model_path = './' + tgt_data + '_checkpoint/' + model + '/model/'
    logs = './logs/'
    confidences_path = './confidences/'
    plot_path = './plots/'

config = DefaultConfigs()
