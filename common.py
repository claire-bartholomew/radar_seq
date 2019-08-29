import iris
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as utils

def prep_data_uk(files):
    cubes = iris.load(files)
    cube = cubes[0]/32
    dataset = cube.data

    print(np.shape(dataset))
    dataset = np.stack(np.split(dataset, dataset.shape[0]/4))
    print(np.shape(dataset))

    # Convert to torch tensors
    tensor = torch.stack([torch.Tensor(i) for i in dataset])
    loader = utils.DataLoader(tensor, batch_size=1)

    return loader

def prep_data(files, augment=False):
    cubes = iris.load(files)
    cube = cubes[0]/32
    # Select square area to concentrate on
    cube = cube[:, 500:1780, 200:1480]
    cube_data = cube.data

    # Data augmentation
    if augment == True:
        cube_data1 = cube.data
        cube_data2 = cube_data1.copy()
        cube_data3 = cube_data1.copy()
        cube_data4 = cube_data1.copy()
        for time in range(np.shape(cube_data1)[0]):
            cube_data2[time] = np.rot90(cube_data1[time])
            cube_data3[time] = np.rot90(cube_data2[time])
            cube_data4[time] = np.rot90(cube_data3[time])
            cube_data = np.append(cube_data1, cube_data2, axis=0)
            cube_data = np.append(cube_data, cube_data3, axis=0)
            cube_data = np.append(cube_data, cube_data4, axis=0)

    # Reshape data
    print(np.shape(cube_data))
    split_data_1 = np.stack(np.split(cube_data, cube_data.shape[0]/16))
    print(np.shape(split_data_1))
    split_data_1 = np.stack(np.split(split_data_1, cube_data.shape[1]/128, -2))
    split_data_1 = np.stack(np.split(split_data_1, cube_data.shape[2]/128, -1))
    print(np.shape(split_data_1))
    dataset = split_data_1.reshape(-1,16,128,128)
    print(np.shape(dataset))

    print(dataset.max())
    print(dataset.mean())

    ## Put data in 'normal' range
    #dataset[np.where(dataset > 32)] = 32
    #dataset[np.where(dataset <= 0)] = 0
    #print(dataset.max())
    #print(dataset.mean())

    # Binarise data
    dataset[np.where(dataset > 0)] = 1
    dataset[np.where(dataset <= 0)] = 0
    print(dataset.max())
    print(dataset.mean())

    # Convert to torch tensors
    tensor = torch.stack([torch.Tensor(i) for i in dataset])
    loader = utils.DataLoader(tensor, batch_size=1)

    return loader

