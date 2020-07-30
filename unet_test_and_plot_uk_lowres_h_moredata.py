import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
import iris
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import iris.plot as iplt
import numpy as np
import pdb
import os
import re
import datetime

import Decomposition_2017 as Decomposition

#===============================================================================
def main():

    rainy_dates = ['1215', '1218', '1206'] #'1127','1109'] # '1108', '1109', '1110', '1112', '1113','1120', '1127', '1128', '1129', '1130','1202', '1204', '1205', '1206', '1207', '1208', '1215', '1216','1217', '1218', '1219', '1220', '1221', '1222']

    # List all possible radar files in range and find those that exist #test or train dir
    files_t = [f'/nobackup/sccsb/radar/test/2018{mmdd}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
               for mi in range(0,60,5) for h in range(24) for mmdd in rainy_dates] 

    #files_t = [f'/nobackup/sccsb/radar/2018{mo:02}{d:02}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
    #             for mi in range(0,60,5) for h in range(10,19) for d in range(28,30) for mo in range(7,8)]
    list_test = []
    for file in files_t:
        if os.path.isfile(file):
            list_test.append(file)
    test_loader, cube = prep_data(list_test, 'test') #'train')

    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load('milesial_unet_10ep_0.01lr_new.pt')) #milesial_unet_uk_15ep_0.01lr_h.pt'))
    model.eval()
    show_outputs(model, test_loader, cube)

#===============================================================================
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#=============================================================================
def prep_data(files, folder):

    # Regrid to a resolution x4 lower
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    timeformat = "%Y%m%d%H%M" # this is how your timestamp looks like
    #regex = re.compile("^/nobackup/sccsb/radar/(\d*)")

    if folder == 'train':
        regex = re.compile("^/nobackup/sccsb/radar/train/(\d*)")
    elif folder == 'test':
        regex = re.compile("^/nobackup/sccsb/radar/test/(\d*)")

    def gettimestamp(thestring):
        m = regex.search(thestring)
        return datetime.datetime.strptime(m.groups()[0], timeformat)

    # sort files by datetime
    sorted_files = sorted(files, key=gettimestamp)

    # only keep filenames where 16 consecutive files exist at 5 min intervals
    sorted_files = list(chunks(sorted_files, 16))
    for group in sorted_files:
        if len(group) < 16:
            sorted_files.remove(group)
        else:
            t0 = group[0].find('2018')
            dt1 = datetime.datetime.strptime(group[0][t0:t0+12], '%Y%m%d%H%M')
            t3 = group[15].find('2018')
            dt2 = datetime.datetime.strptime(group[15][t3:t3+12], '%Y%m%d%H%M')
            if (dt2-dt1 != datetime.timedelta(minutes=75)):
                print(dt2-dt1, 'remove files')
                sorted_files.remove(group)

    dataset = []
    for fn in sorted_files:
        print(fn)
        cube = iris.load_cube(fn)
        cube = cube / 32.
        cube1 = cube.interpolate(sample_points, iris.analysis.Linear())
        data = cube1.data

        # Set limit of large values # or to missing? - have asked Tim Darlington about these large values
        data[np.where(data < 0)] = 0.
        data[np.where(data > 64)] = 64. #-1./32 

        # Normalise data
        data = data / 64.

        # Binarise data 
        #dataset[np.where(dataset < 0)] = 0.
        #dataset[np.where(dataset > 0)] = 1.

        dataset.append(data)
 
    # Convert to torch tensors
    tensor = torch.stack([torch.Tensor(i) for i in dataset])
    loader = utils.DataLoader(tensor, batch_size=1)

    return loader, cube1

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
        print(x2.size(), x1.size())
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
        print('outconv finished')
        return x

#===============================================================================
def show_outputs(net, loader, cube):
    count = 0
    cube = cube[0] # Select just one timestep of cube
    for b, data in enumerate(loader):
        #if ((b > 50) & (b < 100)):
        truth = data[:]
        print(truth, np.shape(truth))
        data = data.type('torch.FloatTensor')
        #Wrap tensors in Variables
        inputs = Variable(data[:,:3])
        # Just test with data with enough rain
        #if ((inputs.mean() > 0.001) & (count<20)):
        count += 1
        #Forward pass
        val_outputs = net(inputs) * 64.

        # Normalise data
        #val_outputs = val_outputs / val_outputs.max()
        print(val_outputs.max())

        val_outputs[np.where(val_outputs < 0.)] = 0.
        #val_outputs[np.where(val_outputs > 32.)] = 32.

        #add to sequence of radar images
        sequence = torch.cat((inputs, val_outputs), 1)

        for step in range(12):
            print('step = {}'.format(step))
            sequence = predict_1hr(sequence, net)

        colors = ['black', 'cornflowerblue', 'royalblue', 'blue', 'lime', 'yellow', 'orange', 'red', 'fuchsia'] #, 'white']
        levels = [0, 0.1, 0.25, 0.5, 1., 2., 4., 8. ,16., 32.]

        for i in range(16):
            print('start figure')
            fig = plt.figure()
            ax = fig.add_subplot(1,2,1)
            cube.data = sequence[0,i].detach().numpy()
            #cf = iplt.contourf(cube, cmap=plt.cm.Blues, vmin=0, vmax=1) #32) #64)
            cf = iplt.contourf(cube, levels, colors=colors)
            cf.cmap.set_over('white')
            plt.gca().coastlines('50m', color='white')
            #plt.setp(ax.xaxis.get_ticklabels(), visible=False)
            #plt.setp(ax.yaxis.get_ticklabels(), visible=False)
            plt.title('U-net t+{}'.format(i))
            #plt.tight_layout()
            #plt.savefig('/home/home01/sccsb/radar_seq/img3/batch{}_im{}.png'.format(b, i))
            #plt.close('all')
            #fig = plt.figure()
            ax = fig.add_subplot(1,2,2)
            cube.data = truth[0,i].detach().numpy()
            cf = iplt.contourf(cube, levels, colors=colors)
            #cf = iplt.contourf(cube, cmap=plt.cm.Blues, vmin=0, vmax=1) #32)
            plt.gca().coastlines('50m', color='white')
            #plt.setp(ax.xaxis.get_ticklabels(), visible=False)
            #plt.setp(ax.yaxis.get_ticklabels(), visible=False)
            plt.title('Truth t+{}'.format(i))
            plt.tight_layout()
            #plt.savefig('/home/home01/sccsb/radar_seq/img3/truth{}_im{}.png'.format(b, i))
            plt.savefig('/home/home01/sccsb/radar_seq/img3/comparison{}_im{}.png'.format(b, i))
            plt.close()

        #elif count >= 100:
        #    break

#===============================================================================
def predict_1hr(sequence, net):
    sequence = sequence.type('torch.FloatTensor')
    inputs = sequence[:,-3:]
    #Wrap tensors in Variables
    inputs = Variable(inputs)
    #Forward pass
    print('forward pass')
    val_outputs = net(inputs) * 64.
    # Normalise data
    #val_outputs = val_outputs / val_outputs.max()
    print(val_outputs.max())
    #val_outputs[np.where(val_outputs > 32.)] = 32.
    val_outputs[np.where(val_outputs < 0.)] = 0.
    print('ouputs calculated')
    sequence = torch.cat((sequence, val_outputs), 1)

    return sequence
#===============================================================================
if __name__ == "__main__":
    main()
