import torch

def Real_AdLoss(discriminator_out, criterion, shape_list):
    # generate ad_label
    ad_label1_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label1 = ad_label1_index.cuda()
    #ad_label1 = ad_label1_index
    ad_label2_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    ad_label2 = ad_label2_index.cuda()
    #ad_label2 = ad_label2_index
    ad_label3_index = torch.LongTensor(shape_list[2], 1).fill_(2)
    ad_label3 = ad_label3_index.cuda()
    #ad_label3 = ad_label3_index
    ad_label4_index = torch.LongTensor(shape_list[3], 1).fill_(3)
    ad_label4 = ad_label4_index.cuda()
    #ad_label4 = ad_label4_index
    ad_label5_index = torch.LongTensor(shape_list[4], 1).fill_(4)
    ad_label5 = ad_label5_index.cuda()
    #ad_label5 = ad_label5_index
    ad_label6_index = torch.LongTensor(shape_list[5], 1).fill_(5)
    ad_label6 = ad_label6_index.cuda()
    #ad_label6 = ad_label6_index
    ad_label = torch.cat([ad_label1, ad_label2, ad_label3, ad_label4, ad_label5, ad_label6], dim=0).view(-1)

    real_adloss = criterion(discriminator_out, ad_label)
    return real_adloss

def Fake_AdLoss(discriminator_out, criterion, shape_list):
    # generate ad_label
    ad_label1_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label1 = ad_label1_index.cuda()
    #ad_label1 = ad_label1_index
    ad_label2_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    ad_label2 = ad_label2_index.cuda()
    #ad_label2 = ad_label2_index
    ad_label3_index = torch.LongTensor(shape_list[2], 1).fill_(2)
    ad_label3 = ad_label3_index.cuda()
    #ad_label3 = ad_label3_index
    ad_label4_index = torch.LongTensor(shape_list[3], 1).fill_(3)
    ad_label4 = ad_label4_index.cuda()
    #ad_label4 = ad_label4_index
    ad_label5_index = torch.LongTensor(shape_list[4], 1).fill_(4)
    ad_label5 = ad_label5_index.cuda()
    #ad_label5 = ad_label5_index
    ad_label6_index = torch.LongTensor(shape_list[5], 1).fill_(5)
    ad_label6 = ad_label6_index.cuda()
    #ad_label6 = ad_label6_index
    ad_label = torch.cat([ad_label1, ad_label2, ad_label3, ad_label4, ad_label5, ad_label6], dim=0).view(-1)

    fake_adloss = criterion(discriminator_out, ad_label)
    return fake_adloss

def AdLoss_Limited(discriminator_out, criterion, shape_list):
    # generate ad_label
    ad_label2_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label2 = ad_label2_index.cuda()
    ad_label3_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    ad_label3 = ad_label3_index.cuda()
    ad_label = torch.cat([ad_label2, ad_label3], dim=0).view(-1)

    real_adloss = criterion(discriminator_out, ad_label)
    return real_adloss
