import gp_emu_uqsa.design_inputs as _gd
import gp_emu_uqsa._emulatorclasses as __emuc
import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.colors as _colors


## checks emulators have been trained previously, and returns minmax of inputs
def emulsetup(emuls):
    minmax = {} # fetch minmax information from the beliefs files
    orig_minmax = {} # fetch minmax information from the beliefs files
    for e in emuls:
        try:
            ai = e.beliefs.active_index
            mm = e.beliefs.input_minmax
        except AttributeError as e:
            print("ERROR: Emulator(s) were not previously trained and reconstructed "
                  "using updated beliefs files, "
                  "so they are missing 'active_index' and 'input_minmax'. Exiting.")
            exit()

        for i in range(len(ai)):
            ## scale minmax into appropriate range
            minmax[str(ai[i])] = list( (_np.array(mm[i]) - mm[i][0])/(mm[i][1] - mm[i][0]) )
            orig_minmax[str(ai[i])] = list( (_np.array(mm[i])) )

    print("\nminmax for active inputs:" , minmax)
    print("original units minmax for active inputs:", orig_minmax)
    return minmax, orig_minmax


## generate sets from active_index inputs
def make_sets(ai):
    sets = []
    for i in ai:
        for j in ai:
            if i!=j and i<j and [i,j] not in sets:
                sets.append([i,j])
    return sets


## reference active indices to ordered list of integers
def ref_act(minmax):
    act_ref = {}
    count = 0
    for key in sorted(minmax.keys(), key=lambda x: int(x)):
        act_ref[key] = count
        count = count + 1
    print("\nrelate active_indices to integers:" , act_ref)
    return act_ref


## may be used later when we restrict which active indices to pair in imp/odp plots
def ref_plt(act):
    plt_ref = {}
    count = 0
    for key in sorted(act):
        plt_ref[str(key)] = count
        count = count + 1
    print("\nrelate restricted active_indices to subplot indices:" , plt_ref)
    return plt_ref


## if user supplies list of active_indices to plot, this checks the list is approp
def check_act(act, sets):
    ## check 'act' is appropriate
    if type(act) is not list:
        print("ERROR: 'act' argument must be a list, but", act, "was supplied. Exiting.")
        exit()
    for a in act:
        if a not in [item for sublist in sets for item in sublist]:
            print("ERROR: index", a, "in 'act' is not an active_index of the emulator(s). Exiting.")
            exit()
    return True


################################
## Plotting utility functions ##
################################

def my_grey():
    return '#696988'

def imp_colormap():
    return _colors.LinearSegmentedColormap.from_list('imp', 
                                        [(0,    '#90ff3c'),
                                         (0.50, '#ffff3c'),
                                         (0.80, '#e2721b'),
                                        (1,    '#db0100')], N=256)

def odp_colormap():
    return _colors.LinearSegmentedColormap.from_list('odp', 
                                        [(0,    my_grey()),
                                         (1.0/float(256),    '#ffffff'),
                                         (0.20, '#93ffff'),
                                         (0.45, '#5190fc'),
                                         (0.65, '#0000fa'),
                                        (1,    '#db00fa')], N=256)

def colormap(cmap, b, t):
    n = 100
    cb   = _np.linspace(b, t, n)
    new_cmap = _colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=b, b=t), cmap( cb ) )
    return new_cmap


def plot_options(plt_ref, ax, fig, minmax=None):

    ## can set labels on diagaonal
    for key in plt_ref:
        ax[plt_ref[key],plt_ref[key]].set(adjustable='box-forced', aspect='equal')
        if minmax is not None:
            ax[plt_ref[key],plt_ref[key]].text(.25,.5,"Input " + str(key) + "\n"
               + str(minmax[key][0]) + "\n-\n" + str(minmax[key][1]))
        fig.delaxes(ax[plt_ref[key],plt_ref[key]]) # for deleting the diagonals

    ## can remove ticks using something like this    
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect('equal')

    ## set the ticks on the edges
    #for i in range(len(minmax)):
    #    for j in range(len(minmax)):
    #        if i != len(minmax) - 1:
    #            ax[i,j].set_xticks([])
    #        if j != 0:
    #            ax[i,j].set_yticks([])
        
    _plt.tight_layout()
    return None


