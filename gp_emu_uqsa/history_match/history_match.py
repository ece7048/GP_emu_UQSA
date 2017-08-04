import gp_emu_uqsa.design_inputs as _gd
import gp_emu_uqsa._emulatorclasses as emuc
import numpy as _np
from scipy import linalg
import matplotlib.pyplot as _plt
from ._hmutilfunctions import *
import pickle
import scipy.spatial.distance as _dist
import diversipy as dv


## using the information in the emulator files, generate an appropriate first design
def first_design(emuls, n, filename = "design.npy"):

    minmax, orig_minmax = emulsetup(emuls)
    act_ref = ref_act(minmax)
    dim = len(act_ref)
    #print("\nDIM:", dim)
    print("\nMINMAX:", minmax)
    print("\nORIG_MINMAX:", orig_minmax)

    olhc_range = [it[1] for it in sorted(minmax.items(), key=lambda x: int(x[0]))]
    #print("\nOLHC:", olhc_range)

    design = _gd.optLatinHyperCube(dim, n, 1, olhc_range, "blank", save = False)

    # if filename supplied, trying memmap...
    if filename != None:
        print("\nNumpy 'save' to file LHC design for HM...")
        _np.save(filename, design)

    return design


## using the information in the emulator files, generate an appropriate first design
def load_design(n, chunks = 1, k = 1, filename = "design.npy"):

    print("Loading chunk", k, "of LHC design...")
    design = _np.load(filename, mmap_mode='r')  # leaves array on disk
    
    ## access chunks of array
    lower, upper = int(k * n/chunks), int((k+1) * n/chunks)
    return design[lower:upper,:]


class Wave:
    """Stores data for wave of HM search."""
    def __init__(self, emuls, zs, cm, var, tests):
        ## passed in
        self.emuls = emuls
        self.zs, self.var, self.cm = zs, var, cm
        if isinstance(tests, _np.ndarray):
            self.TESTS = tests.astype(_np.float16)
            self.I = _np.empty((self.TESTS[:,0].size,len(self.emuls)),dtype=_np.float16)
        else:
            self.TESTS = []
            self.I = []

        ## could replace these with indices into the test points, rather than store more
        self.NIMP = []  # for storing all found imp points (index into test_points..?)
        self.NIMP_I = []  # for storing all found imp points imp values
        ## create a design to fill NROY space based of found NIMP points
        self.NROY = []  

        ## minmax seems to be the corresponding minmax pairs...
        ## orig_minmax seems to be the pre-scaling minmax values...
        self.minmax, self.orig_minmax = emulsetup(emuls)
        self.act_ref = ref_act(self.minmax)  # refs active indices to ordered integers


    ## pickle a list of relevant data
    def save(self, filename):
        print("Pickling wave data in", filename, "...")
        w = [ self.TESTS, self.I, self.NIMP, self.NIMP_I ]  # test these 3 for now
        with open(filename, 'wb') as output:
            pickle.dump(w, output, pickle.HIGHEST_PROTOCOL)
        return

    ## unpickle a list of relevant data
    def load(self, filename):
        print("Unpickling wave data in", filename, "...")
        with open(filename, 'rb') as input:
            w = pickle.load(input)
        self.TESTS, self.I, self.NIMP, self.NIMP_I = w[0], w[1], w[2], w[3]
        return

    ## search through the test inputs to find non-implausible points
    def calc_imps(self):

        P = self.TESTS[:,0].size
        print("\nCalculating Implausibilities of", P ,"points...")
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
            #### K** is 1 - estimation not prediction?

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

                ## calculate implausibility^2 values for point p, output o
                self.I[p,o] = _np.sqrt( ( pmean - z )**2 / ( pvar + v ) )

                ## compare results from different precisions -> differences seem unimportant
                #print("test point:", p, "I:", I[p,o])
                #I2_16[p,o] = ( pmean - z )**2 / ( pvar + v )
                #I2_32[p,o] = ( pmean - z )**2 / ( pvar + v )
                #print("  diff:", I2_32[p,o] - I2_16[p,o])
                
        return

    ## find all the non-implausible points in the test points
    def find_NIMP(self, maxno=1):

        self.NIMP = []  # make empty because may call twice (different maxnos)
        self.NIMP_I = []  # make empty because may call twice (different maxnos)

        ## SHOULD THROW ERROR IF WE HAVEN'T CALCULATED IMP VALUES YET

        P = self.TESTS[:,0].size
        for r in range(P):
            ## find maximum implausibility across different outputs
            Imaxes = _np.sort(_np.partition(self.I[r,:],-maxno)[-maxno:])[-maxno:]
            ## check cut-off
            if Imaxes[-(maxno)] < self.cm:
                self.NIMP.append(self.TESTS[r])
                self.NIMP_I.append(Imaxes)

        self.NIMP = _np.asarray(self.NIMP)
        self.NIMP_I = _np.asarray(self.NIMP_I)
        print("NIMP fraction:", 100*float(len(self.NIMP))/float(P), "%")

        return


## implausibility and optical depth plots for all pairs of active indices
def plot_imps(waves, maxno=1, grid=10, imp_cb=[], odp_cb=[], linewidths=0.2, filename="hexbin.pkl"):

    print("HM plotting. Determining max", maxno,"imps...")

    ## single wave
    if not isinstance(waves, list):
        wave = waves
        TESTS = wave.TESTS
        P = TESTS[:,0].size
        Imaxes = _np.array( [_np.sort(_np.partition(wave.I[r,:],-maxno)[-maxno:])[-maxno]
                             for r in range(P)] )
    ## multi wave
    else:
        wave = waves[0]

        ## combine wave data
        TESTS = wave.TESTS  # first wave's tests
        I = wave.I  # first wave's imps
        for subwave in waves[1:]:
            TESTS = _np.concatenate((TESTS, subwave.TESTS))
            I = _np.concatenate((I, subwave.I))

        P = TESTS[:,0].size
        Imaxes = _np.array( [_np.sort(_np.partition(I[r,:],-maxno)[-maxno:])[-maxno]
                             for r in range(P)] )

    ## space for all plots, and reference index to subplot indices
    print("Creating HM plot objects...")
    rc = len(wave.act_ref)
    fig, ax = _plt.subplots(nrows = rc, ncols = rc)

    ## set colorbar bounds
    imp_cb = [0,wave.cm] if imp_cb == [] else imp_cb
    odp_cb = [0,1] if odp_cb == [] else odp_cb

    ## create list of all pairs of active inputs
    sets = make_sets( [ wave.act_ref[key] for key in wave.act_ref.keys() ] )
    #print("SETS:", sets)

    print("Making subplots of paired indices...")
    ## loop over plot_bins()
    for s in sets:
        ail = [wave.act_ref[str(l)] for l in [s[0], s[1]]]
        ex = ( wave.minmax[str(s[0])][0], wave.minmax[str(s[0])][1],
               wave.minmax[str(s[1])][0], wave.minmax[str(s[1])][1] )

        ax[ail[1],ail[0]].patch.set_facecolor(my_grey())
        im_imp = ax[ail[1],ail[0]].hexbin(
          TESTS[:,ail[0]], TESTS[:,ail[1]], C = Imaxes,
          gridsize=grid, cmap=imp_colormap(), vmin=imp_cb[0], vmax=imp_cb[1],
          extent=ex,
          reduce_C_function=_np.min, linewidths=linewidths, mincnt=1)

        ax[ail[0],ail[1]].patch.set_facecolor(my_grey())
        im_odp = ax[ail[0],ail[1]].hexbin(
          TESTS[:,ail[0]], TESTS[:,ail[1]], C = Imaxes<wave.cm,
          gridsize=grid, cmap=odp_colormap(), vmin=odp_cb[0], vmax=odp_cb[1],
          extent=ex,
          linewidths=linewidths, mincnt=1)

        _plt.colorbar(im_imp, ax=ax[ail[1],ail[0]])
        _plt.colorbar(im_odp, ax=ax[ail[0],ail[1]])

    ## calls to make plot
    plot_options(wave.act_ref, ax, fig, wave.minmax)
    
    ## test pickling of plot
    pickle.dump([fig,ax, imp_cb], open(filename, 'wb'))
    # This is for Python 3 - py2 may need `file` instead of `open`

    _plt.show()

    return


## replot imp/odp using pickle file
def replot_imps(filename="hexbin.pkl", points=[], Is=[]):

    fig, ax, imp_cb = pickle.load(open(filename,'rb'))

    if points != []:
        sets = []
        p = points[:,0].size
        dim = points[0,:].size
        for i in range(dim):
            for j in range(dim):
                if i!=j and i<j and [i,j] not in sets:
                    sets.append([i,j])
        #print("SETS:", sets)

        ## for visualising new wave sim inputs, there will be an option to plot points
        for s in sets:
            for i in range(p):        
                ax[s[1],s[0]].scatter(points[i,s[0]], points[i,s[1]], s=15, c='black')
                ax[s[0],s[1]].scatter(points[i,s[0]], points[i,s[1]], s=15, c='black')
                #ax[s[1],s[0]].scatter(points[i,s[0]], points[i,s[1]], s=20,\
                #        c=Is[i], cmap=imp_colormap(), vmin=imp_cb[0], vmax=imp_cb[1])
                #ax[s[0],s[1]].scatter(points[i,s[0]], points[i,s[1]], s=20,\
                #        c=Is[i], cmap=imp_colormap(), vmin=imp_cb[0], vmax=imp_cb[1])

    _plt.show()

    return


## choose new simulation inputs
def new_inputs(waves, n, N=100):

    print("Calculating spread of new sim points from NIMP...")

    ## single wave
    if not isinstance(waves, list):
        wave = waves
        NIMP = wave.NIMP
        NIMP_I = wave.NIMP_I
        P = NIMP[:,0].size
    ## multi wave
    else:
        wave = waves[0]
        ## combine wave data
        NIMP = wave.NIMP  # first wave's tests
        NIMP_I = wave.NIMP_I
        for subwave in waves[1:]:
            NIMP = _np.concatenate((NIMP, subwave.NIMP))
            NIMP_I = _np.concatenate((NIMP_I, subwave.NIMP_I))
        P = NIMP[:,0].size
 
#    for k in range(0,N):
#        idx = _np.random.choice(NIMP.shape[0], n, replace=False)
#        x = NIMP[idx, :]
#        maximin = _np.argmin( _dist.pdist(x, 'sqeuclidean') )
#        if k==0 or maximin > best_maximin:
#            best_D = _np.copy(x)
#            best_D_I = _np.copy(NIMP_I[idx])
#            best_k = k
#            best_maximin = maximin
#
#    D, D_I, best_k = (best_D, best_D_I, best_k) if N > 1 else (x, 1)
#    if N > 1:  print("Optimal subset was no." , best_k)#, " with D:\n" , D)
#
#    return D, D_I

    #subset = dv.psa_select(NIMP, n, selection_target='centroid_of_hypercube')
    subset = dv.select_greedy_maximin(NIMP, n) ## selects spread out points
    #subset = dv.select_greedy_maxisum(NIMP, n) ## selects extremal points
    #subset = dv.select_greedy_energy(NIMP, n, exponent=int(NIMP[0].size*20)) ## selects spread out points

    return subset


## rescale inputs back into original units
def orig_units(waves, points):

    ## single wave
    if not isinstance(waves, list):
        wave = waves
    ## multi wave
    else:
        wave = waves[0]

    minmax = wave.minmax
    print("MINMAX:", minmax)
    minmax = [it[1] for it in sorted(minmax.items(), key=lambda x: int(x[0]))]
    print("MINMAX:", minmax)
    if wave.emuls[0].all_data.scaled == True:
        print("Unscaling points into original units")
        for i in range(0, points[0].size):
            points[:,i] = points[:,i] \
              * (minmax[i][1] - minmax[i][0]) \
              + minmax[i][0]
    
    return points

