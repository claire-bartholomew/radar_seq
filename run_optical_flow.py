#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.utils.data as utils
#from torch.autograd import Variable
import iris
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
import Decomposition_2017 as Decomposition 

#===============================================================================
def main():

    #files_v = [f'/nobackup/sccsb/radar/20180727{h:02}{m:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
    #             for m in range(0,60,5) for h in range(13,17)]

    files_v = []
    
    for h in range(13,14):
        for m in range(0,60,5):
            files_v.append('/scratch/cbarth/radar/20180727{:02d}{:02d}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(h,m))

    val_loader = prep_data(files_v)

    show_outputs(val_loader)

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

    ## Convert to torch tensors
    #tensor = torch.stack([torch.Tensor(i) for i in dataset])
    #loader = utils.DataLoader(tensor, batch_size=1)

    loader = dataset

    return loader

#===============================================================================
def show_outputs(loader):
    count = 0
    for b, data in enumerate(loader):
        inputs = data
        print(np.shape(data))
        # Just test with data with enough rain
        if ((inputs.mean() > 0.001) & (count<100)):
            sequence = []
            count += 1
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
                plt.tight_layout()
                plt.savefig('optical_flow{}_im{}.png'.format(b, i))
                plt.close('all')

        elif count >= 100:
            break

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
if __name__ == "__main__":
    main()
