import scipy.linalg
import numpy as np
import numpy.ma as ma
import copy
import scipy.signal
import scipy.ndimage.filters
import scipy.ndimage

class OFC(object):
    """
    class to calculate the advection velocity field between two 2D input fields
    There are no specific size constraints on data1 and data2.
    On initialization, the u- and v- component of the advection velocity field
    are calculated, given the two input fields.
    The following routines are defined in the class:
    * ofc.movewholefield(field, ufield=None, vfield=None,dt=1, method=2)
        it advects the input "field" by "dt" time steps with the supplied
        "ufield" and "vfield", or with its own velocity fields that were
        calculated on initialization. If method=1 is chosen, instead of the
        backward advection scheme, as implemented in steps, a forward scheme
        is used.
    """
    def __init__(self, data1=None, data2=None, kernel=7, myboxsize=30,
                 iterations=100, smethod=2, pointweight=0.1, asinsteps=False):
        '''
        input:
            data1: 2D np.array
                first time slice
            data2: 2D np.array
                second time slice
            kernel: integer
                kernel size (radius) for use to smooth the data for partial
                derivative estimaten. If box smoothing is used, half box size
            myboxsize: integer
                side length of box size in which points are evaluated to have
                the same velocity
            iterations: integer
                number of smart smoothing iterations to perform
            smethod: 1 or 2
                smoothing method to be used for smoothing for the calculation
                of the partial derivatives, 1 is box smoothing as in steps
                2 is kernel smoothing
            pointweight: float
                weight given to the velocity of the point (box)
                when doing the smart smoothing. 0.1 is the original steps value
            asinsteps: logical
                if true, smart smoothing 1:1 implementation of steps. If False,
                faster method used (for smart smoothing, this concerns the
                edges)
        '''
        # need to calculate a suitable kernel size given dt (what is expected
        # movement between slices)
        # and expected velocities
        if data1 is not None:
            d1n2 = self.smoothing(data1, kernel, method=smethod)
            d2n2 = self.smoothing(data2, kernel, method=smethod)
            xdif_t = self.totx(d1n2, d2n2, 0)
            ydif_t = self.totx(d1n2, d2n2, 1)
            tdif_t = self.totx(d1n2, d2n2, 2)
            self.ucomp, self.vcomp = self.makeofc(
                xdif_t, ydif_t, tdif_t, myboxsize, d1n2, d2n2, iterations,
                pweight=pointweight, asinsteps=asinsteps)
        else:
            self.ucomp = None
            self.vcomp = None

    @staticmethod
    def mdiffx(xfield):
        """
        This function implements
        x_i+1,j+1 = 0.5[(x_i+1,j - x_i,j) + (x_i+1,j+1 -x_i+1,j+1)]
        """
        d_1 = np.diff(xfield, axis=0)
        d_2 = d_1[:, 1:]
        d_1 = d_1[:, :-1]
        d_3 = np.zeros(xfield.shape)
        d_3[1:, 1:] = (d_1 + d_2)*0.5
        return d_3

    def mdifftx(self, xfield, yfield):
        """
        This function makes (time) average of the x derivative (averaged in y),
        where the two input fields are time slices. Returned is time average
        """
        d_1 = self.mdiffx(xfield)
        d_2 = self.mdiffx(yfield)
        return (d_1 + d_2)*0.5

    def mdiffty(self, xfield, yfield):
        """
        This function returns analogous to mdifftx time averaged y derivative
        (averaged in x) where the two input fields are the time slices.
        """
        x_1 = xfield.transpose()
        y_1 = yfield.transpose()
        intm = self.mdifftx(x_1, y_1)
        return intm.transpose()

    @staticmethod
    def mdifftt(xfield, yfield):
        """ this corresponds to Eq. 31 - 34 in steps document """
        d_1 = yfield - xfield
        d_2 = d_1[1:, :-1]
        d_3 = d_1[:-1, 1:]
        d_4 = d_1[1:, 1:]
        d_1 = d_1[:-1, :-1]
        d_5 = np.zeros(xfield.shape)
        d_5[1:, 1:] = (d_1+d_2+d_3+d_4)*0.25
        return d_5

    def totx(self, xfield, yfield, xty):
        """ this corresponds to Eq. 35 - 37 in steps document """
        if xty == 0:  # this is for x derivative
            m_1 = self.mdifftx(xfield, yfield)
        elif xty == 1:  # this is for y derivative
            m_1 = self.mdiffty(xfield, yfield)
        elif xty == 2:  # this is for t derivative
            m_1 = self.mdifftt(xfield, yfield)
        m_0 = np.zeros([xfield.shape[0]+1, xfield.shape[1]+1])
        m_0[:-1, :-1] = m_1
        m_2 = m_0[1:, :-1]
        m_3 = m_0[:-1, 1:]
        m_4 = m_0[1:, 1:]
        m_1 = m_0[:-1, :-1]
        return (m_1+m_2+m_3+m_4)*0.25

    @staticmethod
    def makeIandDmat(d_u, d_v, d_t):
        """ this corresponds to Eq. 17 in steps document """
        d_u = d_u.flatten()
        d_v = d_v.flatten()
        d_t = d_t.flatten()
        II_mat = (np.array([d_u, d_v])).transpose()
        return II_mat, d_t

    @staticmethod
    def calcU(II_mat, dd):
        """ this corresponds to Eq. 19 in steps document """
        # there is a problem here if I have many fewer pixels with intensity
        # in here than pixels with
        dd = dd.reshape([dd.size, 1])
        IItrans = II_mat.transpose()
        m_1 = IItrans.dot(II_mat)
        m1inv = np.linalg.inv(m_1)
        m_2 = IItrans.dot(dd)
        myreturn = -m1inv.dot(m_2)
        return myreturn[0, 0], myreturn[1, 0]

    @staticmethod
    def makesubboxes(xfield, yfield, tfield, d_1, d_2, boxsize):
        """
        This function makes a list of subboxes with dimension boxsize*boxsize
        of inputfields x,y and t.
        It returns the lists of these subboxes, where the faster varying index
        is the second index in the original fields.
        The original input fields are used to calculate the weight for each
        such subbox.
        """
        xdif_tb = []
        ydif_tb = []
        tdif_tb = []
        weight_tb = []
        for startx in range(0, xfield.shape[0], boxsize):
            for starty in range(0, xfield.shape[1], boxsize):
                xdif_tb.append(
                    xfield[startx:startx+boxsize, starty:starty+boxsize])
                ydif_tb.append(
                    yfield[startx:startx+boxsize, starty:starty+boxsize])
                tdif_tb.append(
                    tfield[startx:startx+boxsize, starty:starty+boxsize])
                newweight = ((
                    (d_1[startx:startx+boxsize, starty:starty+boxsize]).sum() +
                    (d_2[startx:startx+boxsize, starty:starty+boxsize]).sum())
                    / boxsize/boxsize/2.0)
                # this is the weight used in steps
                newweight = 1.0-np.exp(-newweight/0.8)
                weight_tb.append(newweight)
        weight_tb = np.array(weight_tb)
        weight_tb[weight_tb < 0.01] = 0
        return xdif_tb, ydif_tb, tdif_tb, weight_tb

    @staticmethod
    def rebinvel(xfield, yfield, boxsize, myshape, origshape):
        """
        in: xfield, yfield  : u and v box velocity fields
            boxsize   : side length in pixels of each velocity box
            myshape   : shape of the field in velocity box units
            origshape : shape of the field in pixels
        out:  umat_t  : u velocity field
            vmat_t  : v velocity field
        Function to reshape the velocity vectors containing the box velocities,
        to velocity pixel maps
        """
        umat_t = np.zeros(origshape)
        vmat_t = np.zeros(origshape)
        for ii in range(myshape[0]):
            for jj in range(myshape[1]):  # size limited to origshape
                umat_t[ii*boxsize:(ii+1)*boxsize,
                       jj*boxsize:(jj+1)*boxsize] = xfield[ii, jj]
                vmat_t[ii*boxsize:(ii+1)*boxsize,
                       jj*boxsize:(jj+1)*boxsize] = yfield[ii, jj]
        return umat_t, vmat_t

    @staticmethod
    def makekernel(msize):
        """ make a kernel to smooth the input fields """
        temp = 1 - np.abs(np.linspace(-1, 1, msize*2+1))
        kernel = temp.reshape(msize*2+1, 1) * temp.reshape(1, msize*2+1)
        kernel /= kernel.sum()   # kernel should sum to 1!
        return kernel

    def smoothing(self, d_x, sidel, method=2):
        '''
        smoothing used to apply on the field to estimate partial derivatives
        '''
        if method == 1:
            kernel = self.makekernel(sidel)
            dxn = scipy.signal.convolve2d(d_x, kernel, mode='same',
                                          boundary="symm")
        elif method == 2:  # type of smoothing used in steps
            dxn = scipy.ndimage.filters.uniform_filter(d_x, size=sidel*2+1,
                                                       mode='nearest')
        return dxn

    @staticmethod
    def setupweightgrid(field):
        '''
        setup a weightfield:
        0 ---------------0
        |2.5 4 ---- 4 2.5|
        |4   6 -----6   4|
        ||   |      |   ||
        ||   |      |   ||
        |4   6 -----6   4|
        |2.5 4------4 2.5|
        0----------------0
        '''
        xdim, ydim = field.shape+np.array([2, 2])
        zz = np.ones([xdim, ydim])
        zz[0, :] = 0
        zz[:, 0] = 0
        zz[-1, :] = 0
        zz[:, -1] = 0  # halo points
        zz = zz * 6.  # field points
        zz[1, 1:-1] = 4.
        zz[1:-1, 1] = 4.
        zz[-2, 1:-1] = 4.
        zz[1:-1, -2] = 4  # edge points
        zz[1, 1] = 2.5
        zz[-2, -2] = 2.5
        zz[1, -2] = 2.5
        zz[-2, 1] = 2.5  # corner ps
        return zz, xdim, ydim

    @staticmethod
    def makehalo(field):
        '''
        input: 2D array with xdim x ydim
        output: 2D array with xdim+2 x ydim+2, original field with 0 halo
        as:
        0 ----- 0
        | field |
        0 ----- 0
        '''
        halofield = np.zeros(field.shape+np.array([2, 2]))
        halofield[1:-1, 1:-1] = field
        return halofield

    @staticmethod
    def smallkernel():
        '''
        kernel representing the weighting implemented in steps. Used if
        the smartsmooth is used in the convolution mode (i.e. asinsteps=False)
        '''
        mkernel = np.array([[0.5, 1, 0.5], [1, 0, 1], [0.5, 1, 0.5]])/6.
        return mkernel

    def find_neighbour_image(self, field, ww=None):
        '''
        this can replace the cumbersome for loops in the smart smooth.
        However, it does handle edges not exactly the right way.
        From http://stackoverflow.com/questions/22669252/
        how-exactly-does-the-re%E2%80%8C%E2%80%8Bflect-mode-for-
        scipy%E2%80%8C%E2%80%8Bs-ndimage-filters-wo%E2%80%8C%E2%80%8Brk:

        mode       |   Ext   |         Input          |   Ext
        -----------+---------+------------------------+---------
        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        'reflect'  | 3  2  1 | 1  2  3  4  5  6  7  8 | 8  7  6
        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3
        Hence, the mode that corresponds most closely to the for loop solution
        is "mirror" (or reflect) reflect is the default, I leave it with that).
        '''
        mkernel = self.smallkernel()
        if ww is None:
            wfield = field
        else:
            wfield = field*ww
        ofield = scipy.ndimage.convolve(wfield, mkernel)
        return ofield

    def smartsmooth(self, xo, yo, x, y, w, pweight, doasinsteps=False):
        """
        implements the smart smoothing (i.e. 0 that are 0 because there is no
        structure to calculate the advection from, are not included in the
        smoothing of the velocity field) as in steps
        """
        xnew = np.zeros(x.shape)
        ynew = np.zeros(y.shape)
        wnew = np.zeros(w.shape)
        xsm = np.zeros(x.shape)
        ysm = np.zeros(y.shape)
        nweight = 1.0 - pweight
        # should not be done, expensive. The alternative version using
        # a kernel for the neigbours is only slightly different at the edges
        if doasinsteps:
            zz, xdim, ydim = self.setupweightgrid(x)
            xx = self.makehalo(x)
            yy = self.makehalo(y)
            ww = self.makehalo(w)
            for ii in range(1, xdim-1):
                for jj in range(1, ydim-1):
                    xnew[ii-1, jj-1] = (
                        (xx[ii+1, jj]*ww[ii+1, jj]+xx[ii-1, jj]*ww[ii-1, jj]
                         + xx[ii, jj+1]*ww[ii, jj+1]+xx[ii, jj-1]*ww[ii, jj-1])
                        / zz[ii, jj] +
                        (xx[ii+1, jj+1]*ww[ii+1, jj+1]+xx[ii-1, jj+1] *
                         ww[ii-1, jj+1] + xx[ii-1, jj-1]*ww[ii-1, jj-1] +
                         xx[ii+1, jj-1] * ww[ii+1, jj-1]) / (2.*zz[ii, jj]))
                    xsm[ii-1, jj-1] = (
                        (xx[ii+1, jj] + xx[ii-1, jj]+xx[ii, jj+1]+xx[ii, jj-1])
                        / zz[ii, jj]
                        + (xx[ii+1, jj+1]+xx[ii-1, jj+1]+xx[ii-1, jj-1]
                           + xx[ii+1, jj-1]) / (2.*zz[ii, jj]))
                    ysm[ii-1, jj-1] = (
                        (yy[ii+1, jj]+yy[ii-1, jj]+yy[ii, jj+1]+yy[ii, jj-1])
                        / zz[ii, jj]
                        + (yy[ii+1, jj+1] + yy[ii-1, jj+1] + yy[ii-1, jj-1]
                           + yy[ii+1, jj-1]) / (2.*zz[ii, jj]))
                    ynew[ii-1, jj-1] = (
                        (yy[ii+1, jj]*ww[ii+1, jj]+yy[ii-1, jj]*ww[ii-1, jj]
                         + yy[ii, jj+1]*ww[ii, jj+1]+yy[ii, jj-1]*ww[ii, jj-1])
                        / zz[ii, jj]
                        + (yy[ii+1, jj+1]*ww[ii+1, jj+1]+yy[ii-1, jj+1] *
                           ww[ii-1, jj+1]+yy[ii-1, jj-1]*ww[ii-1, jj-1] +
                           yy[ii+1, jj-1]*ww[ii+1, jj-1]) / (2.*zz[ii, jj]))
                    wnew[ii-1, jj-1] = (
                        (ww[ii+1, jj] + ww[ii-1, jj]+ww[ii, jj+1]+ww[ii, jj-1])
                        / zz[ii, jj] +
                        (ww[ii+1, jj+1] + ww[ii-1, jj+1] + ww[ii-1, jj-1] +
                         ww[ii+1, jj-1]) / (2.*zz[ii, jj]))
        else:
            # this does basically the stuff above, except for the borders,
            # those are slightly different, but differences are minimal.
            xnew = self.find_neighbour_image(x, w)
            ynew = self.find_neighbour_image(y, w)
            wnew = self.find_neighbour_image(w)
            xsm = self.find_neighbour_image(x)
            ysm = self.find_neighbour_image(y)

        # xsm will stay as the normal (unweighted average) for all points
        # where the neigbours have no weight and the point has no weight
        # below, everywhere where the neigbours have weight, I replace the
        # value with the weighted average of the neigbours
        xsm[abs(wnew) > 0] = xnew[abs(wnew) > 0]/wnew[abs(wnew) > 0]
        ysm[abs(wnew) > 0] = ynew[abs(wnew) > 0]/wnew[abs(wnew) > 0]
        # if the point itself has weight, I use a weighted sum of the neigbour
        # points and the point (however, it is not the point in iteration, it
        # is the original value!)
        xsm[abs(w) > 0] = (xnew[abs(w) > 0]*nweight + pweight * xo[abs(w) > 0]
                           * w[abs(w) > 0]) / (nweight*wnew[abs(w) > 0] +
                                               pweight*w[abs(w) > 0])
        ysm[abs(w) > 0] = (ynew[abs(w) > 0]*nweight + pweight * yo[abs(w) > 0]
                           * w[abs(w) > 0]) / (nweight*wnew[abs(w) > 0] +
                                               pweight*w[abs(w) > 0])
        return xsm, ysm

    def makeofc(self, xdif_t, ydif_t, tdif_t, subboxsize, d1, d2,
                iterations=50, pweight=0.1, asinsteps=False):
        """
        this implements the OFC algorithm, assuming all points in a box with
        subboxsize sidelength have the same velocity components.
        """
        # This part is for doing it the way it is done in STEPS----------------
        # (a) make subboxes
        # pweight = 0.1 # this is the weight given to the point as opposed to
        # its neigbours
        xdif_tb, ydif_tb, tdif_tb, weight_tb = self.makesubboxes(
            xdif_t, ydif_t, tdif_t, d1, d2, subboxsize)
        uvec = []
        vvec = []
        # (b) solve ofc on subboxes
        for xdif, ydif, tdif in zip(
                xdif_tb, ydif_tb, tdif_tb):
            II, dd = self.makeIandDmat(xdif, ydif, tdif)
            try:
                u = self.calcU(II, dd)
            except:
                u = [0, 0]
            uvec.append(u[0])
            vvec.append(u[1])
        uvec = np.array(uvec)
        vvec = np.array(vvec)
        # print uvec
        weight_tb = np.array(weight_tb)
        # (c) reorder block velocities in 2D
        myshape = [int((xdif_t.shape[0]-1)/subboxsize) + 1,
                   int((xdif_t.shape[1]-1)/subboxsize) + 1]
        umat = uvec.reshape(myshape)
        vmat = vvec.reshape(myshape)
        # pause()
        weights = weight_tb.reshape(myshape)
        # 03.11.2016 build in a check to detect insane velocities and put them
        # to an average of neigbouring
        flag = (np.abs(umat) + np.abs(vmat)) > vmat.shape[0]/3.
        umat[flag] = 0
        vmat[flag] = 0
        weights[flag] = 0
        # (d) do some smart smoothing
        conv_vec = []
        umatn = np.copy(umat)
        vmatn = np.copy(vmat)
        for _ in range(iterations):
            umatold = umatn
            umatn, vmatn = self.smartsmooth(umat, vmat, umatn, vmatn, weights,
                                            pweight, doasinsteps=asinsteps)
            conv_vec.append((abs(umatold-umatn)).sum())
        # (e) rebin block velocities to 2D field velocities
        umat_f, vmat_f = self.rebinvel(umatn, vmatn, subboxsize, myshape,
                                       xdif_t.shape)
        smn = int(subboxsize/3)
        umat_f = self.smoothing(umat_f, smn, method=1)
        vmat_f = self.smoothing(vmat_f, smn, method=1)
        # ---------------------------------------------------------------------
        return umat_f, vmat_f

    def movewholefield(self, field, ufield=None, vfield=None,
                       dt=1, method=2, bgd=0.0):
        '''
        This function moves the whole 2D field by the velocity field
        represented by ufield and vfield, if supplied, or by OFC.ucomp and
        OFC.vcomp. The velocity is assumed to be in cells/timestep. dt is the
        number of time steps to move the field.
        If method=2, a backward advection scheme (as in steps) is used.
        If method is 1, a forward advection scheme is used (be aware of the
        discretization effects). bgd defines the background 0.
        '''
        mfield = np.zeros(field.shape)
        if ufield is None:
            ufield = self.ucomp
        if vfield is None:
            vfield = self.vcomp
        # first check on the correct size:
        if vfield.shape != field.shape:
            try:
                raise TransformError(1, field.shape, vfield.shape)
            except: # TransformError, exc:
                print('TransformError') #exc.message)
        # backward advection: now the assumption is more that the velocity at
        # the point in the field indicates where (in the next time step) the
        # point at the current position came from
        if method == 2:
            ydim, xdim = field.shape
            (ygrid, xgrid) = np.meshgrid(np.arange(xdim),
                                         np.arange(ydim))
            oldx_frac = -ufield * dt + xgrid.astype(float)
            oldy_frac = -vfield * dt + ygrid.astype(float)
            mfield = np.full(field.shape, bgd)
            cond1 = (oldx_frac >= 0.) & (oldy_frac >= 0.) & (
                     oldx_frac < ydim) & (oldy_frac < xdim)
            mfield[cond1] = 0
            oldx_l = oldx_frac.astype(int)
            oldx_r = oldx_l + 1
            x_frac_r = oldx_frac - oldx_l.astype(float)
            oldy_u = oldy_frac.astype(int)
            oldy_d = oldy_u + 1
            y_frac_d = oldy_frac - oldy_u.astype(float)
            cond2 = ((oldx_l >= 0) & (oldy_u >= 0) & (oldx_l < ydim) &
                     (oldy_u < xdim) & cond1)
            cond3 = ((oldx_r >= 0) & (oldy_u >= 0) & (oldx_r < ydim) &
                     (oldy_u < xdim) & cond1)
            cond4 = ((oldx_l >= 0) & (oldy_d >= 0) & (oldx_l < ydim) &
                     (oldy_d < xdim) & cond1)
            cond5 = ((oldx_r >= 0) & (oldy_d >= 0) & (oldx_r < ydim) &
                     (oldy_d < xdim) & cond1)
            for ii, cond in enumerate([cond2, cond3, cond4, cond5], 2):
                xorig = xgrid[cond]
                yorig = ygrid[cond]
                if ii == 2:
                    xfr = 1.-x_frac_r
                    yfr = 1.-y_frac_d
                    xc = oldx_l[cond]
                    yc = oldy_u[cond]
                elif ii == 3:
                    xfr = x_frac_r
                    yfr = 1. - y_frac_d
                    xc = oldx_r[cond]
                    yc = oldy_u[cond]
                elif ii == 4:
                    xfr = 1.-x_frac_r
                    yfr = y_frac_d
                    xc = oldx_l[cond]
                    yc = oldy_d[cond]
                elif ii == 5:
                    xfr = x_frac_r
                    yfr = y_frac_d
                    xc = oldx_r[cond]
                    yc = oldy_d[cond]
                mfield[xorig, yorig] = (
                    mfield[xorig, yorig] + field[xc, yc] *
                    xfr[xorig, yorig]*yfr[xorig, yorig])
        else:  # this is a forward advection
            #  advect field with velocity (ufield,vfield) where u/vfield is
            # in units of cells per dt here u is in direction of 1st dimension
            # This implements a forward model: The contribution of intensity at
            # point i,j at time t+1 is calculated as sum of all contributions
            # from points with intensity that end up at i,j after t+1. This is
            # opposed to the backward implementation of moveholefield2, where
            # intensity at point i,j at time t+1 is the sum of the origins of
            # I(delta_t*-u).
            for ii in range(field.shape[0]):
                for jj in range(field.shape[1]):
                    newx_frac = ufield[ii, jj] * dt + ii
                    newy_frac = vfield[ii, jj] * dt + jj
                    # find the ii and jj values these positions lie in between
                    newx_l = int(newx_frac)
                    newx_r = (newx_l + 1)
                    x_frac_r = newx_frac - float(newx_l)
                    newy_u = int(newy_frac)
                    newy_d = (newy_u + 1)
                    y_frac_d = newy_frac - float(newy_u)
                    try:
                        mfield[newx_l, newy_u] += (
                            1. - x_frac_r) * (1. - y_frac_d) * field[ii, jj]
                    except:
                        pass
                    try:
                        mfield[newx_r, newy_u] += x_frac_r * (
                            1. - y_frac_d) * field[ii, jj]
                    except:
                        pass
                    try:
                        mfield[newx_l, newy_d] += (
                            1. - x_frac_r) * y_frac_d * field[ii, jj]
                    except:
                        pass
                    try:
                        mfield[newx_r, newy_d] += x_frac_r*y_frac_d*field[ii,
                                                                          jj]
                    except:
                        pass
        return mfield

# -----------------------------------------------------------------------------
