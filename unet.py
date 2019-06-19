import iris
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torch.autograd import Variable

#===============================================================================
def main():

    datadir = '/nobackup/sccsb/radar' #/scratch/cbarth/radar

    files = []
    for h in range(7,12):
        for m in range(0,60,5):
            #files.append('{}/20190216{:02d}{:02d}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(datadir, h, m))
            files.append('{}/20180727{:02d}{:02d}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(datadir, h, m))

    train_loader = load_data(files)

    files2 = []
    for h in range(12,17):
        for m in range(0,60,5):
            files2.append('{}/20180727{:02d}{:02d}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(datadir, h, m))

    val_loader = load_data(files2)

    files3 = []
    for h in range(17,22):
        for m in range(0,60,5):
            files3.append('{}/20180727{:02d}{:02d}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(datadir, h, m))

    test_loader = load_data(files3)

    unet = UNet()
    val_loss, len_val_data = train_net(unet, train_loader, val_loader, batch_size=100,
                                       n_epochs=10, learning_rate=0.01)
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in unet.state_dict():
        print(param_tensor, "\t", unet.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    torch.save(unet.state_dict(), 'unet_model.pt')

    show_outputs(unet, test_loader)

#===============================================================================
def load_data(files):
    '''Load radar data and split into 128x128 2d grid domains as multiple
       sequences of 4 timesteps'''

    cubes = iris.load(files)

    cube = cubes[0]/32
    #pdb.set_trace()
    cube = cube[:, 500:1780, 200:1480]
    #cube = cube[:10*(cube.shape[0]//10), 1000:3560, 1000:2280]
    cube_data1 = cube.data
    print(np.shape(cube_data1))

    # rotate and append data for data augmentation
    cube_data2 = cube_data1.copy()
    cube_data3 = cube_data1.copy()
    for time in range(np.shape(cube_data1)[0]):
        cube_data2[time] = np.rot90(cube_data1[time])
        cube_data3[time] = np.rot90(cube_data2[time])

    cube_data = np.append(cube_data1, cube_data2, axis=0)
    cube_data = np.append(cube_data, cube_data3, axis=0)

    split_data_1 = np.stack(np.split(cube_data, cube_data.shape[0]/4))
    print(np.shape(split_data_1))
    split_data_1 = np.stack(np.split(split_data_1, cube_data.shape[1]/128, -2))
    split_data_1 = np.stack(np.split(split_data_1, cube_data.shape[2]/128, -1))
    print(np.shape(split_data_1))
    dataset = split_data_1.reshape(-1,4,128,128)
    print(np.shape(dataset))
    print(dataset.max())
    print(dataset.mean())

    # Binarise dataset
    dataset[np.where(dataset > 0)] = 1
    dataset[np.where(dataset <= 0)] = 0
    print(dataset.max())
    print(dataset.mean())


    # Convert to torch tensors
    tensor = torch.stack([torch.Tensor(i) for i in dataset])
    train_loader = utils.DataLoader(tensor, batch_size=1)

    return train_loader

#===============================================================================
def show_outputs(net, loader):
    for b, data in enumerate(loader):
        if ((b > 50) & (b < 100)):
            data = data.type('torch.FloatTensor')
            inputs, labels = data[:,:3], data[:,3]
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            #Forward pass
            val_outputs = net(inputs)

            #re-binarise output
            val_outputs[np.where(val_outputs < 0.2)] = 0

            #add to sequence of radar images
            sequence = torch.cat((inputs, val_outputs), 1)

            for step in range(12):
                sequence = sequence.type('torch.FloatTensor')
                #inputs, labels = sequence[:,-4:-1], sequence[:,-1]
                inputs = sequence[:,-3:]
                #Wrap tensors in Variables
                inputs = Variable(inputs)
                #Forward pass
                val_outputs = net(inputs)
                val_outputs[np.where(val_outputs < 0.2)] = 0

                sequence = torch.cat((sequence, val_outputs), 1)

            for i in range(12):
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                #ax = fig.add_subplot(2, 6, i+1)
                cf = plt.contourf(sequence[0,i].detach().numpy(), cmap=plt.cm.Greys)
                ax.set_xticks(np.arange(0, 128, 10))
                ax.set_yticks(np.arange(0, 128, 10))
                plt.grid()
                plt.setp(ax.xaxis.get_ticklabels(), visible=False)
                plt.setp(ax.yaxis.get_ticklabels(), visible=False)
                plt.savefig('plot_batch{}_im{}.png'.format(b, i))
                plt.close()

#===============================================================================
def createLossAndOptimizer(net, learning_rate=0.01):

    #Loss function
    loss = torch.nn.MSELoss()

    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return(loss, optimizer)

#===============================================================================
def train_net(net, train_loader, val_loader, batch_size, n_epochs, learning_rate):

    #Print the hyperparameters of the training run:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    #Get training data
    n_batches = len(train_loader)

    #Create the loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    #Time for printing
    training_start_time = time.time()

    loss_list = []

    #Loop for n_epochs
    for epoch in range(n_epochs):

        start_time = time.time()
        running_loss = 0.0
        print_every = n_batches // 10
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):

            #Get inputs from training data
            inputs, labels = data[:,:3], data[:,3]

            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)

            # Run the forward pass https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
            outputs = net(inputs)
            loss_size = loss(outputs[0], labels)
            loss_list.append(loss_size.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad() #set param gardients to zero
            loss_size.backward()
            optimizer.step()

            #print(loss_size.data.item(), loss_size.item())

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, n_epochs, i + 1, n_batches, loss_size.item()))

            #Print statistics
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()

            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t training loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every,
                        time.time() - start_time))

                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for data in val_loader:
            inputs, labels = data[:,:3], data[:,3]
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs[0], labels)
            total_val_loss += val_loss_size.data.item()
            #print(val_loss_size, total_val_loss, len(val_loader))

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader))) #check this is printing what we expect

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    return(total_val_loss, len(val_loader))

#===============================================================================
class double_conv(nn.Module):
    '''(conv => ReLU) * 2''' # from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down_conv(nn.Module):
    '''maxpool => (conv => relu) * 2'''
    def __init__(self, in_ch, out_ch):
        super(down_conv, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up_conv(nn.Module):
    '''convT => concat => (conv => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_ch//2, in_ch//2, kernel_size=2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        #print(x1.size())
        #print(x2.size())
        x = torch.cat([x2, x1], dim=1)
        #print(x.size())
        x = self.conv(x)
        #print(x.size())
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()

        self.conv = double_conv(in_ch, out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.inc = inconv(3, 16)
        self.down1 = down_conv(16, 32)
        self.down2 = down_conv(32, 64)
        self.down3 = down_conv(64, 128)
        self.down4 = down_conv(128, 128)
        self.up4 = up_conv(256, 64)
        self.up3 = up_conv(128, 32)
        self.up2 = up_conv(64, 16)
        self.up1 = up_conv(32, 16)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        orig = x
        #print('x=', x.size())
        x1 = self.inc(x)
        size1 = x1.size()
        #print('x1=', x1.size())
        x2 = self.down1(x1)
        #print('x2=', x2.size())
        x3 = self.down2(x2)
        #print('x3=', x3.size())
        x4 = self.down3(x3)
        #print('x4=', x4.size())
        x5 = self.down4(x4)
        #print('x5=', x5.size())
        x = self.up4(x5, x4)
        #print('x=', x.size())
        #print(x3.size())
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.conv2(x)

        return(x)

#===============================================================================
if __name__ == "__main__":
    main()
