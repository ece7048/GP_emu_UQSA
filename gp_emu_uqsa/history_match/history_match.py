# this is attempt to redo history matching
        ## minmax seems to be the corresponding minmax pairs...
        ## orig_minmax seems to be the pre-scaling minmax values...

import gp_emu_uqsa.design_inputs as _gd
import gp_emu_uqsa._emulatorclasses as emuc
import numpy as _np
from scipy import linalg
import matplotlib.pyplot as _plt
from ._hmutilfunctions import *


## using the information in the emulator files, generate an appropriate first design
def first_design(emuls, n):

    minmax, orig_minmax = emulsetup(emuls)
    act_ref = ref_act(minmax) ## references the active indices to an ordered list of integers

    dim = len(act_ref)

    print("\nDIM:", dim)
    print("\nMINMAX:", minmax)

    olhc_range = [it[1] for it in sorted(minmax.items(), key=lambda x: int(x[0]))]

    print("\nOLHC:", olhc_range)

    design = _gd.optLatinHyperCube(dim, n, 1, olhc_range, "blank", save = False)

    return design


class Wave:
    """Stores data for wave of HM search."""
    def __init__(self, emuls, zs, cm, var, tests):
        ## non-data stuff IN
        self.emuls = emuls
        self.zs = zs
        self.var = var
        self.cm = cm
        ## data stuff IN
        self.TESTS = tests
        # float16 saves memory
        self.I = _np.empty((self.TESTS[:,0].size,len(self.emuls)),dtype=_np.float16)
        ## to be calculated
        self.NIMP = []  # for storing all found imp points (index into test_points..?)
        self.NROY_filler = []  # create a design to fill NROY space based of found NIMP points
        self.NROY_design = []  # representative inputs from NROY space - could do outside...

        #### WE MAY BE ABLE TO COMBINE THESE NOW WE DON'T MAKE AN oLHC HERE
        #### BUT MAY BE NEEDED FOR PRODUCING THE NEW WAVE FILLER LATER
        ## minmax seems to be the corresponding minmax pairs...
        ## orig_minmax seems to be the pre-scaling minmax values...
        self.minmax, self.orig_minmax = emulsetup(emuls)
        self.act_ref = ref_act(self.minmax) ## references the active indices to an ordered list of integers


    ## search through the test inputs to find non-implausible points
    def calc_imps(self):

        print("\nCalculating Implausibilities...")
        P = self.TESTS[:,0].size
        #I2_16 = _np.empty((P,len(self.emuls)),dtype=_np.float16) # could even use float16..?
        #I2_32 = _np.empty((P,len(self.emuls)),dtype=_np.float32) # could even use float16..?

        ## loop over outputs (i.e. over emulators)
        for o in range(len(self.emuls)):
            E, z, v = self.emuls[o], self.zs[o], self.var[o]
            Eai = E.beliefs.active_index

            ## extract the active inputs for this emulator from test inputs
            print("Active inputs for this emul:", Eai)
            act_ind_list = [self.act_ref[str(l)] for l in Eai]
            print("Matching indices in design array:", act_ind_list)

            ## NOTES
            #### K** is 1 - does this mean estimation not prediction?

            Hnew = _np.empty([len(E.training.basis.h)]) 
            beta = E.training.par.beta
            K = E.training.A
            y = E.training.outputs
            Hold = E.training.H
            s2 = E.training.beliefs.sigma**2  # sigma*2
            Knew = 1.0 # K**
            invA_H = linalg.solve( K, Hold )
            Q = Hold.T.dot(invA_H)  # H A^-1 H
            T = linalg.solve(K, y - Hold.dot(beta))

            #L = np.linalg.cholesky(self.data.A) 
            #w = np.linalg.solve(L,self.data.H)
            #Q = w.T.dot(w) # H A^-1 H

            ## single point test for now...
            #x = self.TESTS[0, act_ind_list]
            #if True:

            print("Calculating implausibilites for output", o, "...")
            ## loop over test points
            for p in range(P):
                x = self.TESTS[p, act_ind_list]

                covar = E.K.covar(E.training.inputs, x.reshape(x.shape[0],-1).T)  # K*

                for j in range(0, len(E.training.basis.h)):
                    Hnew[j] = E.training.basis.h[j](x)  # H*

                R = Hnew - (covar.T).dot(invA_H)

                pmean = Hnew.dot(beta) + covar.T.dot(T)
                pvar  = s2*( Knew - covar.T.dot( linalg.solve( K, covar ) ) )

                ## calculate implausibility^2 values for each output
                self.I[p,o] = _np.sqrt( ( pmean - z )**2 / ( pvar + v ) )
                #print("test point:", p, "I:", I[p,o])
                #I2_16[p,o] = ( pmean - z )**2 / ( pvar + v )
                #I2_32[p,o] = ( pmean - z )**2 / ( pvar + v )
                #print("  diff:", I2_32[p,o] - I2_16[p,o])
                
        return


    def find_NIMP(self, cm, maxno):

        self.NIMP = []  # make it empty again because we may call twice..?

        ## SHOULD THROW ERROR IF WE HAVEN'T CALCULATED IMP VALUES YET

        P = self.TESTS[:,0].size

        #Imaxes = _np.empty([P,maxno], dtype=_np.float16)
        for r in range(P):
            #Imaxes[r,:] = _np.sort(_np.partition(self.I[r,:],-maxno)[-maxno:])[-maxno:]
            ## find maximum implausibility across different outputs
            Imaxes = _np.sort(_np.partition(self.I[r,:],-maxno)[-maxno:])[-maxno:]
            if Imaxes[-(maxno)] < cm: # check cut-off using this value
                self.NIMP.append(self.TESTS[r])
        self.NIMP = _np.asarray(self.NIMP)
        #print("NIMP points:", self.NIMP)
        print("NIMP fraction:", 100*float(len(self.NIMP))/float(P), "%")

        return


    def plot_bins(self, cm, maxno, aiX, aiY, grid=10):
        
        P = self.TESTS[:,0].size

        IMP = _np.full( (grid,grid), 10000. )
        ODP = _np.zeros( (grid,grid,2) )  # third dim is pass (index 0) or fail index (1)

        ## find minX, maxX, dX etc.
        (minX, maxX) = self.minmax[str(aiX)]
        (minY, maxY) = self.minmax[str(aiY)]
        dX = (maxX-minX)/float(grid)
        dY = (maxY-minY)/float(grid)
        ail = [self.act_ref[str(l)] for l in [aiX, aiY]]

        for r in range(P):
            ## find maximum implausibility across different outputs
            Imaxes = _np.sort(_np.partition(self.I[r,:],-maxno)[-maxno:])[-maxno]

            # calc. I and J index for bins
            I = int( (self.TESTS[r,ail[0]] - minX)/dX )
            J = int( (self.TESTS[r,ail[1]] - minY)/dY )

            if Imaxes < cm:
                ODP[I,J,0] += 1  # increase pass number
            else:
                ODP[I,J,1] += 1  # increase fail number

            # check if less implausible than current min
            if Imaxes < IMP[I,J]:  IMP[I,J] = Imaxes

        for I in range(grid):
            for J in range(grid):
                ODP[I,J,0] = ODP[I,J,0] / ( ODP[I,J,0] + ODP[I,J,1] )

        return IMP, ODP[:,:,0]


    def big_plot_bins(self, cm, maxno, grid=10, INTERP='none'):

        ## space for all plots, and reference index to subplot indices
        print("Creating plot objects... may take some time...")
        rc = len(self.act_ref)
        fig, ax = _plt.subplots(nrows = rc, ncols = rc)

        ## create list of all pairs of active inputs
        sets = make_sets( [ self.act_ref[key] for key in self.act_ref.keys() ] )
        print("SETS:", sets)

        imp_cb = [0,cm]
        odp_cb = [0,1]

        ## loop over plot_bins()
        for s in sets:
            IMP, ODP = self.plot_bins(cm, maxno, s[0], s[1], grid)
            make_plots(s, self.act_ref, cm, maxno, ax, IMP, ODP, minmax=self.minmax, imp_cb=imp_cb, odp_cb=odp_cb, INTERP=INTERP)

        ## calls to make plot
        plot_options(self.act_ref, ax, fig, self.minmax)
        _plt.show()

        return 0


    def big_plot_hexbins(self, cm, maxno, grid=10, INTERP='none'):

        ## space for all plots, and reference index to subplot indices
        print("Creating plot objects... may take some time...")
        rc = len(self.act_ref)
        fig, ax = _plt.subplots(nrows = rc, ncols = rc)

        ## create list of all pairs of active inputs
        sets = make_sets( [ self.act_ref[key] for key in self.act_ref.keys() ] )
        print("SETS:", sets)

        imp_cb = [0,cm]
        odp_cb = [0,1]

        P = self.TESTS[:,0].size
        Imaxes = _np.array( [_np.sort(_np.partition(self.I[r,:],-maxno)[-maxno:])[-maxno]
                             for r in range(P)] )

        ## loop over plot_bins()
        for s in sets:
            ail = [self.act_ref[str(l)] for l in [s[0], s[1]]]
            ex = ( self.minmax[str(s[0])][0], self.minmax[str(s[0])][1],
                   self.minmax[str(s[1])][0], self.minmax[str(s[1])][1] )

            im_imp = ax[ail[1],ail[0]].hexbin(
              self.TESTS[:,ail[0]], self.TESTS[:,ail[1]], C = Imaxes,
              gridsize=grid, cmap=imp_colormap(), vmin=imp_cb[0], vmax=imp_cb[1],
              reduce_C_function=_np.min)

            im_odp = ax[ail[0],ail[1]].hexbin(
              self.TESTS[:,ail[0]], self.TESTS[:,ail[1]], C = Imaxes<cm,
              gridsize=grid, cmap=odp_colormap(), vmin=odp_cb[0], vmax=odp_cb[1])

            _plt.colorbar(im_imp, ax=ax[ail[1],ail[0]])
            _plt.colorbar(im_odp, ax=ax[ail[0],ail[1]])

            #make_plots(s, self.act_ref, cm, maxno, ax, IMP, ODP, minmax=self.minmax, imp_cb=imp_cb, odp_cb=odp_cb, INTERP=INTERP)

        ## calls to make plot
        plot_options(self.act_ref, ax, fig, self.minmax)
        _plt.show()

        return 0


def prev_imp_plot(emuls, zs, cm, var_extra, maxno=1, olhcmult=100, grid=10, act=[], fileStr="", plot=True):
    """Create an implausibility and optical depth plot, made of subplots for each pair of active inputs (or only those specified). Implausibility plots in the lower triangle, optical depth plots in the upper triangle. The diagonal is blank, and implausibility plots are paired with optical depth plots across the diagonal.

    Args:
        emuls (Emulator list): list of Emulator instances
        zs (float list): list of output values to match
        cm (float list): cut-off for implausibility
        var_extra (float list): extra (non-emulator) variance on outputs
        maxno (int): which maximum implausibility to consider, default 1
        olhcmult (int): option for size of oLHC design across other inputs not in the considered pair, size = olhcmult*(no. active inputs - 2), default 100
        grid (int): divisions of each input range to make, with values of each input for a subplot centred on the gridpoint, default 10
        act (int list): list of active inputs for plot, default [] (all inputs)
        fileStr (str): string to prepend to output files, default ""
        plot (bool): choice to plot (e.g. False for batches), default True

    Returns:
        None

    """

    sets, minmax, orig_minmax = emulsetup(emuls)
    check_act(act, sets)
    act_ref = ref_act(minmax)
    plt_ref = ref_plt(act)

    num_inputs = len(minmax) # number of inputs we'll look at
    dim = num_inputs - 2 # dimensions of input that we'll change with oLHC

    maxno=int(maxno)
    IMP , ODP = [], [] ## need an IMP and ODP for each I_max
    for i in range(maxno):
        IMP.append( _np.zeros((grid,grid)) )
        ODP.append( _np.zeros((grid,grid)) )

    ## space for all plots, and reference index to subplot indices
    print("Creating plot objects... may take some time...")
    plot = True if plot == True else False
    rc = num_inputs if act == [] else len(act)
    if plot:
        fig, ax = _plt.subplots(nrows = rc, ncols = rc)
    plot_ref = act_ref if act == [] else ref_plt(act)

    ## reduce sets to only the chosen ones
    less_sets = []
    if act == []:
        less_sets = sets
    else:
        for s in sets:
            if s[0] in act and s[1] in act:
                less_sets.append(s)
    print("HM for input pairs:", less_sets)

    return 0
