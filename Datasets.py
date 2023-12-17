import math
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import hickle
import time

def data_generator(dataset, batch_size, dataroot, shuffle=True):


    if dataset == "TUH":

        train_X_train = np.load ("/file.npy")
        train_y_train = np.load ("/file.npy")

        ones_indices = np.where(train_y_train == 1)[0] 

        final_data = train_X_train[ones_indices]
        final_data_Y = train_y_train [ones_indices]

        duplicated_ones_indices = np.repeat(final_data_Y,4)
        duplicated_train_X_train = np.repeat(final_data,4,axis= 0)

        train_X_train = np.concatenate ((train_X_train,duplicated_train_X_train),axis=0)
        train_y_train = np.concatenate ((train_y_train,duplicated_ones_indices),axis=0)

        train_y_train = train_y_train.astype(np.int64)

        print (train_X_train.shape)
        print (train_y_train.shape)

        print("Number of 1s:", np.count_nonzero(train_y_train == 1))
        print("Number of 0s:", np.count_nonzero(train_y_train == 0))

        test_X_train = np.load("/yikai.npy")
        test_y_train = np.load("/yikai.npy")

        print("Number of 1s:", np.count_nonzero(test_y_train == 1))
        print("Number of 0s:", np.count_nonzero(test_y_train == 0))

        test_y_train = test_y_train.astype(np.int64)

        train_X_train = np.transpose(train_X_train, (0, 1, 3, 2))  
        test_X_train = np.transpose(test_X_train, (0, 1, 3, 2))

        train_dataset = TensorDataset(torch.FloatTensor(train_X_train), torch.tensor(train_y_train))
        test_dataset = TensorDataset(torch.FloatTensor(test_X_train), torch.tensor(test_y_train))

        train_loader = DataLoader(train_dataset,batch_size=batch_size,
        shuffle=True,generator=torch.Generator(device='cuda'))

        test_loader = DataLoader(test_dataset,batch_size=batch_size,
        shuffle=False,generator=torch.Generator(device='cuda'))

        n_classes = 1
        seq_length = 23*125
        input_channels = 19

    else:
        print('Please provide a valid dataset name.')
        exit(1)

    return train_loader, test_loader, seq_length, input_channels, n_classes

def extra_test_generator (patname, year, batch_size):


    test_X_train = np.load("/file.npy")[:]
    test_y_train = np.load("/file.npy")[:]

    #TUH,validation

    # patient_folder_path = "/file/path"
    # file_list = [file for file in os.listdir(patient_folder_path) if file.endswith('.npy')]
    # concatenated_data = []whe
    # test_y_train = np.zeros(len(file_list))
    #
    # for i, file_name in enumerate(file_list):
    #
    #     file_path = os.path.join(patient_folder_path, file_name)
    #     data = np.load(file_path)
    #     concatenated_data.append(data)
    #     print(i)
    #
    #     if 'bckg' not in file_name:
    #         test_y_train[i] = 1

    # test_X_train = np.stack(concatenated_data, axis=0)

    test_X_train = test_X_train.astype(np.float16)

    print("Number of 1s:", np.count_nonzero(test_y_train == 1))
    print("Number of 0s:", np.count_nonzero(test_y_train == 0))

    test_y_train = test_y_train.astype(np.int64)

    test_X_train = np.transpose(test_X_train, (0, 2, 1, 3))

    print (test_X_train.shape)
    print (test_y_train.shape)

    test_dataset = TensorDataset(torch.FloatTensor(test_X_train), torch.tensor(test_y_train))

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, generator=torch.Generator(device='cuda'))

    return test_loader

def Epilepsia_12s_STFT (patname,batch_size):

    TestX = hickle.load ("/hickle")
    TestY = hickle.load ("/hickle")

    TestX1 = np.concatenate([TestX[0][i] for i in range(len(TestX[0]))], axis=0)
    print ("done1")
    TestX2 = np.concatenate([TestY[0][i] for i in range(len(TestY[0]))], axis=0)
    print ("done2")
    TestX3 = np.concatenate([TestX[1][i] for i in range(len(TestX[0]))], axis=0)
    print ("done3")
    TestX4 = np.concatenate([TestY[1][i] for i in range(len(TestY[0]))], axis=0)
    print ("done4")

    TestX = np.concatenate((TestX1,TestX2), axis=0)
    TestY = np.concatenate((TestX3,TestX4), axis=0)

    print (TestX.shape)
    print (TestY.shape)

    TestX = TestX.astype(np.float16)
    test_X_train = np.transpose(TestX, (0, 2, 1, 3))
    test_X_train = test_X_train[:, :, :, :125]
    test_y_train = TestY.astype(np.int64)

    print("Number of 1s:", np.count_nonzero(test_y_train == 1))
    print("Number of 0s:", np.count_nonzero(test_y_train == 0))

    test_dataset = TensorDataset(torch.FloatTensor(test_X_train), torch.tensor(test_y_train))

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, generator=torch.Generator(device='cuda'))

    return test_loader


