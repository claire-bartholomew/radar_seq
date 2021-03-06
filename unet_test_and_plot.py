import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
import iris
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb

import Decomposition_2017 as Decomposition 

#===============================================================================
def main():

    #files_v = [f'/s3/mo-uki-radar-comp/20180915{h:02}{m:02}_nimrod_ng_radar_rainrate_composite_500m_UK' \
    #           for m in range(0,60,5) for h in range(6,8)]
    files_v = [f'/nobackup/sccsb/radar/20180727{h:02}{m:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
                 for m in range(0,60,5) for h in range(13,17)]

    val_loader = prep_data(files_v)
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load('milesial_unet_model.pt'))
    model.eval()
    show_outputs(model, val_loader)

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
def run_optical_flow(haic_t1, haic_t2, haic_t3):

    print('Running optical flow method...')
    # Determine velocity fields using optical flow method inputs: (inputfield1,
    # inputfield2, timestepBetweenFields, Kernelwidth,
    # BoxsizeOfBoxestoHaveSameVelocity, NumberOfIterations, smoothingmethod,
    # pointweight)  ##inputfields are data arrays

    #Determine optical flow class (calculates the advection velocity field between two 2D input fields)
    myNewofc1 = Decomposition.OFC(haic_t1, haic_t2, kernel=7,
                                  myboxsize=3, iterations=20, smethod=1,
                                  pointweight=0.1)
    myNewofc2 = Decomposition.OFC(haic_t2, haic_t3, kernel=7,
                                  myboxsize=3, iterations=20, smethod=1,
                                  pointweight=0.1)

    #Calculate average velocity fields from the 2 timesteps
    average_ofc_t0 = myNewofc2
    average_ofc_t0.ucomp = (myNewofc1.ucomp + myNewofc2.ucomp)/2.
    average_ofc_t0.vcomp = (myNewofc1.vcomp + myNewofc2.vcomp)/2.

    #The calculated velocity components are stored in myNewofc.ucomp and myNewofc.vcomp
    #Apply velocity field to inputfield2 using a backward in time scheme to
    #minimize artefacts. A forward in time scheme can be used by setting the
    #optional argument method=1.
    advected_field_t0 = average_ofc_t0.movewholefield(haic_t3)

    myNewofc3 = Decomposition.OFC(haic_t3, advected_field_t0, kernel=7,
                                  myboxsize=3, iterations=20, smethod=1,
                                  pointweight=0.1)

    #Calculate average velocity fields from the 2 timesteps
    average_ofc_t1 = myNewofc3
    average_ofc_t1.ucomp = (myNewofc2.ucomp + myNewofc3.ucomp)/2.
    average_ofc_t1.vcomp = (myNewofc2.vcomp + myNewofc3.vcomp)/2.

    advected_field_t1 = average_ofc_t1.movewholefield(advected_field_t0)

    myNewofc4 = Decomposition.OFC(advected_field_t0, advected_field_t1,
                                  kernel=7, myboxsize=3, iterations=20,
                                  smethod=1, pointweight=0.1)

    #Calculate average velocity fields from the 2 timesteps
    average_ofc_t2 = myNewofc4
    average_ofc_t2.ucomp = (myNewofc3.ucomp + myNewofc4.ucomp)/2.
    average_ofc_t2.vcomp = (myNewofc3.vcomp + myNewofc4.vcomp)/2.

    advected_field_t2 = average_ofc_t2.movewholefield(advected_field_t1)

    return advected_field_t0, advected_field_t1, advected_field_t2

#===============================================================================
def plot_optical_flow(inputs, b):
    inputs = inputs.numpy()
    inputs = inputs[0]
    print(inputs)
    print(np.shape(inputs))
    sequence = []
    t1 = inputs[0]
    t2 = inputs[1]
    t3 = inputs[2]
    fcst1, fcst2, fcst3 = run_optical_flow(t1, t2, t3)
    fcst4, fcst5, fcst6 = run_optical_flow(fcst1, fcst2, fcst3)
    fcst7, fcst8, fcst9 = run_optical_flow(fcst4, fcst5, fcst6)
    fcst10, fcst11, fcst12 = run_optical_flow(fcst7, fcst8, fcst9)
    fcst13, fcst14, fcst15 = run_optical_flow(fcst10, fcst11, fcst12)

    for im in [t1, t2, t3, fcst1, fcst2, fcst3, fcst4, fcst5, fcst6, fcst7, fcst8, fcst9, fcst10, fcst11, fcst12, fcst13, fcst14, fcst15]:
        sequence.append(im)
        #sequence = sequence[0]

    for i in range(14):
        print(sequence[i], np.shape(sequence[i]))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        cf = plt.contourf(sequence[i], cmap=plt.cm.Greys)
        plt.title('Optical flow timestep: {}'.format(i))
        plt.tight_layout()
        plt.savefig('optical_flow{}_im{}.png'.format(b, i))
        plt.close('all')

#===============================================================================
def show_outputs(net, loader):
    count = 0
    for b, data in enumerate(loader):
        #if ((b > 50) & (b < 100)):
        truth = data[:]
        print(truth, np.shape(truth))
        data = data.type('torch.FloatTensor')
        #Wrap tensors in Variables
        inputs = Variable(data[:,:3])
        # Just test with data with enough rain
        if ((inputs.mean() > 0.001) & (count<20)):

            # Run data through optical flow methods
            plot_optical_flow(inputs, b)

            count += 1
            #Forward pass
            val_outputs = net(inputs)

            #re-binarise output
            val_outputs[np.where(val_outputs < 0.2)] = 0

            #add to sequence of radar images
            sequence = torch.cat((inputs, val_outputs), 1)

            for step in range(1, 16):
                sequence = sequence.type('torch.FloatTensor')
                inputs = sequence[:,-3:]
                #Wrap tensors in Variables
                inputs = Variable(inputs)
                #Forward pass
                val_outputs = net(inputs)
                val_outputs[np.where(val_outputs < 0.2)] = 0

                sequence = torch.cat((sequence, val_outputs), 1)

            for i in range(16):
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                cf = plt.contourf(sequence[0,i].detach().numpy(), cmap=plt.cm.Greys)
                #ax.set_xticks(np.arange(0, 128, 10))
                #ax.set_yticks(np.arange(0, 128, 10))
                #plt.grid()
                #plt.setp(ax.xaxis.get_ticklabels(), visible=False)
                #plt.setp(ax.yaxis.get_ticklabels(), visible=False)
                plt.title('U-net timestep: {}'.format(i))
                plt.tight_layout()
                plt.savefig('/home/home01/sccsb/radar_seq/img/batch{}_im{}.png'.format(b, i))
                plt.close('all')
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                cf = plt.contourf(truth[0,i].detach().numpy(), cmap=plt.cm.Greys)
                #ax.set_xticks(np.arange(0, 128, 10))
                #ax.set_yticks(np.arange(0, 128, 10))
                #plt.grid()
                #plt.setp(ax.xaxis.get_ticklabels(), visible=False)
                #plt.setp(ax.yaxis.get_ticklabels(), visible=False)
                plt.title('Observed timestep: {}'.format(i))
                plt.tight_layout()
                plt.savefig('/home/home01/sccsb/radar_seq/img/truth{}_im{}.png'.format(b, i))
                plt.close()

        elif count >= 100:
            break
#===============================================================================
if __name__ == "__main__":
    main()
