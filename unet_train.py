import argparse
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

import radar_seq.common as common
import radar_seq.unet_model as model

#===============================================================================
def main(nepochs, lr):

    print(nepochs, lr)

    files_t = [f'/nobackup/sccsb/radar/201807{d:02}{h:02}{m:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
               for m in range(0,60,5) for h in range(6,9) for d in range(27,31)]
    train_loader = common.prep_data(files_t)

    files_v = [f'/nobackup/sccsb/radar/201807{d:02}{h:02}{m:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
             for m in range(0,60,5) for h in range(10,13) for d in range(27,31)]
    val_loader = common.prep_data(files_v)

    unet = model.UNet(n_channels=3, n_classes=1)

    trained_net = train_net(unet, train_loader, val_loader,
                            batch_size=100, n_epochs=nepochs, learning_rate=lr)
    torch.save(trained_net.state_dict(), 'milesial_unet_{}ep_{}lr.pt'.format(str(nepochs), str(lr)))


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

    return(net)


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
