import iris
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torch.optim as optim
import time
from torch.autograd import Variable

#===============================================================================
def main():

    files_t = [f'/nobackup/sccsb/radar/20180727{h:02}{m:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
               for m in range(0,60,5) for h in range(6,9)]
    train_loader = prep_data(files_t)

    files = [f'/nobackup/sccsb/radar/20180727{h:02}{m:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
             for m in range(0,60,5) for h in range(10,13)]
    val_loader = prep_data(files_v)

    unet = UNet(n_channels=3, n_classes=1)
    val_loss, len_val_data = train_net(unet, train_loader, val_loader,
                                batch_size=100, n_epochs=15, learning_rate=0.01)

#===============================================================================
def prep_data(files):
    cubes = iris.load(files)
    cube = cubes[0]/32
    # Select square area to concentrate on
    cube = cube[:, 500:1780, 200:1480]
    #cube = cube[:10*(cube.shape[0]//10), 1000:2280, 1000:2280]
    cube_data1 = cube.data

    # Data augmentation
    cube_data2 = cube_data1.copy()
    cube_data3 = cube_data1.copy()
    for time in range(np.shape(cube_data1)[0]):
        cube_data2[time] = np.rot90(cube_data1[time])
        cube_data3[time] = np.rot90(cube_data2[time])

        cube_data = np.append(cube_data1, cube_data2, axis=0)
        cube_data = np.append(cube_data, cube_data3, axis=0)

    # Reshape data
    print(np.shape(cube_data))
    split_data_1 = np.stack(np.split(cube_data, cube_data.shape[0]/4))
    print(np.shape(split_data_1))
    split_data_1 = np.stack(np.split(split_data_1, cube_data.shape[1]/128, -2))
    split_data_1 = np.stack(np.split(split_data_1, cube_data.shape[2]/128, -1))
    print(np.shape(split_data_1))
    dataset = split_data_1.reshape(-1,4,128,128)
    print(np.shape(dataset))

    print(dataset.max())
    print(dataset.mean())

    # Binarise data
    dataset[np.where(dataset > 0)] = 1
    dataset[np.where(dataset <= 0)] = 0
    print(dataset.max())
    print(dataset.mean())

    # Convert to torch tensors
    tensor = torch.stack([torch.Tensor(i) for i in dataset])
    loader = utils.DataLoader(tensor, batch_size=1)

    return loader

#===============================================================================
# Create model
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

    torch.save(unet.state_dict(), 'milesial_unet_model.pt')

    return(total_val_loss, len(val_loader))

#===============================================================================
# full assembly of the sub-parts to form the complete net
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

#===============================================================================
# sub-parts of the U-Net model
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

#===============================================================================
if __name__ == "__main__":
    main()
