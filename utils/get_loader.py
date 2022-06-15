import os
from torch.utils.data import DataLoader
from utils.dataset import SSDGDataset
from utils.utils import load_directory_files, split_fake_real

def get_dataset(src1_data, src2_data, src3_data, src4_data,
                #src5_data, src6_data,
                #tgt_data,
                batch_size, path):
    print('Load Source Data')
    print('Source Data: ', src1_data)
    #src1_train_features, src1_train_labels = load_files(os.path.join(path, src1_data))
    src1_train_features, src1_train_labels = load_directory_files(path, src1_data)
    src1_train_data_real, src1_train_data_fake = split_fake_real(src1_train_features, src1_train_labels)

    print('Load Source Data')
    print('Source Data: ', src2_data)
    #src2_train_features, src2_train_labels = load_files(os.path.join(path, src2_data))
    src2_train_features, src2_train_labels = load_directory_files(path, src2_data)
    src2_train_data_real, src2_train_data_fake = split_fake_real(src2_train_features, src2_train_labels)

    print('Load Source Data')
    print('Source Data: ', src3_data)
    #src3_train_features, src3_train_labels = load_files(os.path.join(path, src3_data))
    src3_train_features, src3_train_labels = load_directory_files(path, src3_data)
    src3_train_data_real, src3_train_data_fake = split_fake_real(src3_train_features, src3_train_labels)

    print('Load Source Data')
    print('Source Data: ', src4_data)
    #src4_train_features, src4_train_labels = load_files(os.path.join(path, src4_data))
    src4_train_features, src4_train_labels = load_directory_files(path, src4_data)
    src4_train_data_real, src4_train_data_fake = split_fake_real(src4_train_features, src4_train_labels)

    #print('Load Source Data')
    #print('Source Data: ', src5_data)
    #src5_train_features, src5_train_labels = load_files(os.path.join(path, src5_data))
    #src5_train_data_real, src5_train_data_fake = split_fake_real(src5_train_features, src5_train_labels)

    #print('Load Source Data')
    #print('Source Data: ', src6_data)
    #src6_train_features, src6_train_labels = load_files(os.path.join(path, src6_data))
    #src6_train_data_real, src6_train_data_fake = split_fake_real(src6_train_features, src6_train_labels)

    #print('Load Target Data')
    #print('Target Data: ', tgt_data)
    #tgt_test_features, tgt_test_labels = load_files(os.path.join(path, tgt_data))
    #tgt_test_data = [tgt_test_features, tgt_test_labels]

    src1_train_dataloader_fake = DataLoader(SSDGDataset(src1_train_data_fake), batch_size=batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(SSDGDataset(src1_train_data_real), batch_size=batch_size, shuffle=True)

    src2_train_dataloader_fake = DataLoader(SSDGDataset(src2_train_data_fake), batch_size=batch_size, shuffle=True)
    src2_train_dataloader_real = DataLoader(SSDGDataset(src2_train_data_real), batch_size=batch_size, shuffle=True)

    src3_train_dataloader_fake = DataLoader(SSDGDataset(src3_train_data_fake), batch_size=batch_size, shuffle=True)
    src3_train_dataloader_real = DataLoader(SSDGDataset(src3_train_data_real), batch_size=batch_size, shuffle=True)

    src4_train_dataloader_fake = DataLoader(SSDGDataset(src4_train_data_fake), batch_size=batch_size, shuffle=True)
    src4_train_dataloader_real = DataLoader(SSDGDataset(src4_train_data_real), batch_size=batch_size, shuffle=True)

    #src5_train_dataloader_fake = DataLoader(SSDGDataset(src5_train_data_fake), batch_size=batch_size, shuffle=True)
    #src5_train_dataloader_real = DataLoader(SSDGDataset(src5_train_data_real), batch_size=batch_size, shuffle=True)

    #src6_train_dataloader_fake = DataLoader(SSDGDataset(src6_train_data_fake), batch_size=batch_size, shuffle=True)
    #src6_train_dataloader_real = DataLoader(SSDGDataset(src6_train_data_real), batch_size=batch_size, shuffle=True)

    #tgt_dataloader = DataLoader(SSDGDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=False)

    return src1_train_dataloader_fake, src1_train_dataloader_real, \
           src2_train_dataloader_fake, src2_train_dataloader_real, \
           src3_train_dataloader_fake, src3_train_dataloader_real, \
           src4_train_dataloader_fake, src4_train_dataloader_real
           #src5_train_dataloader_fake, src5_train_dataloader_real, \
           #src6_train_dataloader_fake, src6_train_dataloader_real, \
           #tgt_dataloader









