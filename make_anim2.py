import os
from os import listdir
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import pdb
import fnmatch
import matplotlib
import matplotlib.image as mgimg
import matplotlib.animation as animation
import iris
import matplotlib.colors as colors
import iris.quickplot as qplt
import iris.plot as iplt
import pdb

def animate():
    filepath = '/scratch/cbarth/phd/img6/'

    fig = plt.figure(figsize=(18,8))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    # initiate an empty list of "plotted" images
    myimages = []

    # loops through available pngs
    for batch in range(21, 27):
        print('batch = {}'.format(batch))
        for f in range(14): #16):

            ## Read in picture
            fname1 = filepath + 'truth{}_im{}.png'.format(batch, f)
            fname2 = filepath + 'batch{}_im{}.png'.format(batch, f)
            fname3 = filepath + 'optical_flow{}_im{}.png'.format(batch, f)
            img1 = mgimg.imread(fname1)
            img2 = mgimg.imread(fname2)
            img3 = mgimg.imread(fname3)

            #imgplot = plt.imshow(img)
            imgplot1 = ax1.imshow(img1)
            imgplot2 = ax2.imshow(img2)
            imgplot3 = ax3.imshow(img3)
            myimages.append([imgplot1, imgplot2, imgplot3])

            # append AxesImage object to the list
            #myimages.append([imgplot])

        ## create an instance of animation
        my_anim = animation.ArtistAnimation(fig, myimages, interval=500, repeat = True)

        # save animation
        #my_anim.save('truth_anim.mp4')
        #my_anim.save('predict_anim.mp4')
        my_anim.save('all_anim_{}.mp4'.format(batch))

#===============================================================================
def main():

    animate()

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
