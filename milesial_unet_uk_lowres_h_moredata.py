import argparse
import iris
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torch.optim as optim
import time
from torch.autograd import Variable

#===============================================================================
def main(nepochs, lr):
    print(nepochs, lr)
    # List all possible radar files in range and find those that exist
    files_t = [f'/nobackup/sccsb/radar/2018{mo:02}{d:02}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
               for mi in range(0,60,5) for h in range(24) for d in range(5) for mo in range(5,6)] #8)]
    list_train = []
    for file in files_t:
        if os.path.isfile(file):
            list_train.append(file)
    train_loader = prep_data(list_train)

    files_v = [f'/nobackup/sccsb/radar/2018{mo:02}{d:02}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
               for mi in range(0,60,5) for h in range(24) for d in range(25,28) for mo in range(5,6)] #8)]
    list_val = []
    for file in files_v:
        if os.path.isfile(file):
            list_val.append(file)
    val_loader = prep_data(list_val)

    unet = UNet(n_channels=3, n_classes=1)

    trained_net = train_net(unet, train_loader, val_loader,
                            batch_size=100, n_epochs=nepochs, learning_rate=lr)
    torch.save(trained_net.state_dict(), 'milesial_unet_uk_{}ep_{}lr_h2.pt'.format(str(nepochs), str(lr)))

#===============================================================================
def prep_data(files):
    cubes = iris.load(files)
    print('loaded cubes')
    cube = cubes[0]/32

    # Regrid to a resolution x4 lower
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]
    cube1 = cube.interpolate(sample_points, iris.analysis.Linear())

    # Separate into groups of 4 time steps
    dataset = cube1.data
    print(np.shape(dataset))
    dataset = np.stack(np.split(dataset, dataset.shape[0]/4))
    print(np.shape(dataset))

    # Set limit of large values # or to missing? - have asked Tim Darlington about these large values
    dataset[np.where(dataset < 0)] = 0.
    dataset[np.where(dataset > 32)] = 32. #-1./32 

    # Normalise data
    dataset = dataset/32.

    # Binarise data 
    #dataset[np.where(dataset < 0)] = 0.
    #dataset[np.where(dataset > 0)] = 1.

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

            print('inputs = {}'.format(inputs.max()))
            print('outputs = {}'.format(outputs.max()))

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

    #torch.save(net.state_dict(), 'milesial_unet_model.pt')

    return(net) #total_val_loss, len(val_loader))

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
        return x #torch.sigmoid(x)

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
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

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
    # Parse the arguments. The required arguments are the location of
    # interest, the files containing the data to plot, the height
    # at which the wind speed is required, and the map zoom.
    parser = argparse.ArgumentParser(description='Run unet')
    parser.add_argument('-e', type=int, required=True,
                        help='the number of epochs')
    parser.add_argument('-l', type=float, required=True,
                        help='the learning rate')

    args = parser.parse_args()
    n_epochs = args.e
    learningrate = args.l

    main(n_epochs, learningrate)
