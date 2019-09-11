import torch
from torch.autograd import Variable
import iris
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb

import common as common
import unet_model as model

#===============================================================================
def main():

    files_v = [f'/nobackup/sccsb/radar/20180727{h:02}{m:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
                 for m in range(0,60,5) for h in range(13,17)]

    val_loader = common.prep_data(files_v)
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load('milesial_unet_model.pt'))
    model.eval()
    show_outputs(model, val_loader)

#===============================================================================
def show_outputs(net, loader):
    count = 0
    for b, data in enumerate(loader):
        truth = data[:]
        print(truth, np.shape(truth))
        data = data.type('torch.FloatTensor')
        #Wrap tensors in Variables
        inputs = Variable(data[:,:3])
        # Just test with data with enough rain
        if ((inputs.mean() > 0.001) & (count<20)): #limit output to 20 batches

            count += 1
            #Forward pass
            val_outputs = net(inputs)

            #re-binarise output
            #val_outputs[np.where(val_outputs < 0.2)] = 0

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
                plt.title('U-net timestep: {}'.format(i))
                plt.tight_layout()
                plt.savefig('/home/home01/sccsb/radar_seq/img/batch{}_im{}.png'.format(b, i))
                plt.close('all')
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                cf = plt.contourf(truth[0,i].detach().numpy(), cmap=plt.cm.Greys)
                plt.title('Observed timestep: {}'.format(i))
                plt.tight_layout()
                plt.savefig('/home/home01/sccsb/radar_seq/img/truth{}_im{}.png'.format(b, i))
                plt.close()

        elif count >= 100:
            break
#===============================================================================
if __name__ == "__main__":
    main()
