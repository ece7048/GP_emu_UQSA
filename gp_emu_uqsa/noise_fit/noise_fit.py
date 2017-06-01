import gp_emu_uqsa as g
import numpy as np
import gp_emu_uqsa._emulatorclasses as __emuc
import gp_emu_uqsa.design_inputs as _gd


### transform z = log(r)
def __transform(x):
    return np.log(x)

### untransform E[r] = exp(E[z] + V[z]/2)
def __untransform(mean, var):
    return np.exp(mean + 0.5*var)

def __read_file(ifile):
    print("*** Reading file:", ifile ,"***")
    dct = {}
    try:
        with open(ifile, 'r') as f:
            for line in f:
                (key, val) = line.split(' ',1)
                dct[key] = val.strip()
        return dct
    except OSError as e:
        print("ERROR: Problem reading file.")
        exit()


# currently works only for 1D data
def noisefit(data, noise, stopat=20, olhcmult=100, samples=200, fileStr=""):
    """Try to fit one emualtor to the mean of the data and another emulator to the noise of the data. Results of estimating the noise are saved to the files 'noise-inputs' and 'noise-outputs'.

    Args:
        data (str): Name of configuration file for fitting the input-output data.
        noise (str): Name of configuration file for fitting the input-noise.
        stopat (int): Number of iterations.
        olhcmult (int): Scales the number of data points in the results files.

    Returns:
        None

    """

    #### check transform option
    ## if not "log", no transformation will be done
    
    #### check consistency
    datac, noisec = __read_file(data), __read_file(noise)
    datab, noiseb = __read_file(datac["beliefs"]), __read_file(noisec["beliefs"])
    if datac["inputs"] != noisec["inputs"]:
        print("\nWARNING: different inputs files in config files. Exiting.")
        return None 
    if datab["mucm"] == 'T':
        print("\nWARNING: data beliefs must have mucm F, "
              "as sigma (presumably) not valid if extra pointwise variance is added. Exiting.")
        return None
    if datab["fix_nugget"] == 'T' or  noiseb["fix_nugget"] == 'T':
        print("\nWARNING: data and noise beliefs need fix_nugget F. Exiting.")
        return None
    if datac["tv_config"] !=  noisec["tv_config"]:
        print("\nWARNING: different tv_config in config files. Exiting.")
        return None 
    if noisec["outputs"] != "zp-outputs":
        print("\nWARNING: noise config outputs must be 'zp-outputs'. Exiting.")
        return None 

    ## setup emulators here
    GD = g.setup(data, datashuffle=True, scaleinputs=False)
    ## create 'zp-outputs' file with zeros
    np.savetxt("zp-outputs", \
      np.zeros(GD.training.outputs.size + GD.validation.outputs.size*GD.tv_conf.noV).T)
    GN = g.setup(noise, datashuffle=True, scaleinputs=False)

    ## if shuffled, fix the inconsistencies
    GN.training.inputs = GD.training.inputs
    GN.validation.inputs = GD.validation.inputs
    GN.training.remake()
    GN.validation.remake()

    ## if we have validation sets, set no_retrain=True
    if GD.all_data.tv.noV > 1:
        print("\nWARNING: should have 0 or 1 validation sets for noise fitting. Exiting.")
        ## extra validation sets would be totally unused
        exit()
    valsets = False if GD.all_data.tv.noV == 0 else True


    #### step 1 ####
    print("\n****************"
          "\nTRAIN GP ON DATA"
          "\n****************")
    #GD = g.setup(data, datashuffle=False, scaleinputs=False)
    x = GD.training.inputs # values of the inputs
    t = GD.training.outputs # values of the noisy outputs
    if valsets:
        xv = GD.validation.inputs # values of the inputs
        tv = GD.validation.outputs

    #print(np.amin(x), np.amax(x))
    g.train(GD, no_retrain=valsets)

    r = np.zeros(t.size)
    if valsets:
        rv = np.zeros(tv.size)

    ## we stay within this loop until done 'stopat' fits
    count = 0
    while True:
        if count == 0:
            xp = __emuc.Data(x, None, GD.basis, GD.par, GD.beliefs, GD.K)
            if valsets:
                xvp = __emuc.Data(xv, None, GD.basis, GD.par, GD.beliefs, GD.K)
        else:
            #### step 5 - return to step 2 if not converged ####
            xp = __emuc.Data(x, None, GD.basis, GD.par, GD.beliefs, GD.K)
            xp.set_r(r)
            xp.make_A(s2 = GD.par.sigma**2 , predict = True)
            if valsets:
                xvp = __emuc.Data(xv, None, GD.basis, GD.par, GD.beliefs, GD.K)
                xvp.set_r(rv)
                xvp.make_A(s2 = GD.par.sigma**2 , predict = True)
        count = count + 1


        #### step 2 - generate D'={(xi,zi)} ####
        print("\n***********************"
              "\nESTIMATING NOISE LEVELS " + str(count) +
              "\n***********************")

        post = __emuc.Posterior(xp, GD.training, GD.par, GD.beliefs, GD.K, predict = True)
        L = np.linalg.cholesky(post.var)
        z_prime = np.zeros(t.size)
        s = samples
        for j in range(s): # predict 's' different values
            u = np.random.randn(t.size)
            tij = post.mean + L.dot(u)
            z_prime = z_prime + 0.5*(t - tij)**2
        z_prime = __transform(z_prime/float(s))
        np.savetxt('zp-outputs' , z_prime)

        # estimate noise levels for validation set
        if valsets:
            post = __emuc.Posterior(xvp, GD.training, GD.par, GD.beliefs, GD.K, predict = True)
            L = np.linalg.cholesky(post.var)
            z_prime_V = np.zeros(tv.size)
            s = samples
            for j in range(s): # predict 's' different values
                u = np.random.randn(tv.size)
                tij = post.mean + L.dot(u)
                z_prime_V = z_prime_V + 0.5*(tv - tij)**2
            z_prime_V = __transform(z_prime_V/float(s))


        #### step 3 ####
        # train a GP on x and z
        print("\n*****************"
              "\nTRAIN GP ON NOISE " + str(count) +
              "\n*****************")
        ## need to setup again so as to re-read updated zp-outputs
        #GN = g.setup(noise, datashuffle=False, scaleinputs=False)
        #GN.training.outputs = np.loadtxt('zp-outputs').T
        GN.training.outputs = z_prime.T
        GN.training.remake()
        if valsets:
            GN.validation.outputs = z_prime_V.T
            GN.validation.remake()
        ## fix to allow retraining using same training set against validation
        GN.tv_conf.no_of_trains = 0
        GN.tv_conf.retrain = 'y'
        g.train(GN, no_retrain=valsets)


        #### step 4 - use GN to predict noise values for G3 ####
        print("\n***********************************"
              "\nTRAIN GP ON DATA WITH NOISE FROM GP " + str(count) +
              "\n***********************************")

        xp_GN = __emuc.Data(x, None, GN.basis, GN.par, GN.beliefs, GN.K)
        p_GN = __emuc.Posterior(xp_GN, GN.training, GN.par, GN.beliefs, GN.K, predict = True) ## I'VE CHANGED THIS TO FALSE
        r = __untransform(p_GN.mean, np.diag(p_GN.var))
        #r = __untransform(p_GN.mean, 0.0)

        #GD = g.setup(data, datashuffle=False, scaleinputs=False)
        GD.training.set_r(r)

        ## add estimated r to the validation set for better diagnostics
        if valsets:
            v_GN = __emuc.Data(xv, None, GN.basis, GN.par, GN.beliefs, GN.K)
            pv_GN = __emuc.Posterior(v_GN, GN.training, GN.par, GN.beliefs, GN.K, predict = True)
            rv = __untransform(pv_GN.mean, np.diag(pv_GN.var))
            #rv = __untransform(pv_GN.mean, 0.0)
            GD.validation.set_r(rv)

        ## fix to allow retraining using same training set against validation
        GD.tv_conf.no_of_trains = 0
        GD.tv_conf.retrain = 'y'
        g.train(GD, no_retrain=valsets)

        # break when we've done 'stopat' fits
        if count == stopat:
            print("\nCompleted", count, "fits, stopping here.")

            ## use an OLHC design for x_values of noise guesses we'll save
            print("\nGenerating input points to predict noise values at...")
            n = x[0].size * int(olhcmult)
            N = int(n)
            olhc_range = [ [np.amin(col), np.amax(col)] for col in x.T ]
            #print("olhc_range:", olhc_range)
            filename = "x_range_input"
            _gd.optLatinHyperCube(x[0].size, n, N, olhc_range, filename)
            x_range = np.loadtxt(filename) # read generated oLHC file in

            # if 1D inputs, store in 2D array with only 1 column
            if x[0].size == 1:
                x_range = np.array([x_range,]).T

            ## save data to file
            x_plot = __emuc.Data(x_range, None, GN.basis, GN.par, GN.beliefs, GN.K)
            p_plot = __emuc.Posterior(x_plot, GN.training, GN.par, GN.beliefs, GN.K, predict = True)
            mean_plot = p_plot.mean
            var_plot = p_plot.var
            p_plot.interval()
            UI, LI = p_plot.UI, p_plot.LI

            print("\nSaving results to file...")
            nfileStr = fileStr + "_" if fileStr != "" else fileStr
            np.savetxt(nfileStr + 'noise-inputs', x_range )

            #np.savetxt(nfileStr + 'noise-outputs', np.transpose(\
            #  [np.sqrt(__untransform(mean_plot, np.diag(var_plot), reg=REG)),\
            #  np.sqrt(__untransform(LI, 0.0, reg=REG)), np.sqrt(__untransform(UI, 0.0, reg=REG))] ) )

            np.savetxt(nfileStr + 'noise-outputs', np.transpose(\
              [  __untransform(mean_plot, np.diag(var_plot)) ,\
                 __untransform(LI, 0.0) ,\
                 __untransform(UI, 0.0) ] ) )

            break

    return None


## posterior means of mean prediction and noise prediction
def noisepost(data, noise, X_inputs):

    #### load config files for data and noise

    ## note that inputs and outputs are already in same order for both
    GD = g.setup("config-data-recon",  datashuffle = False, scaleinputs = False)
    GN = g.setup("config-noise-recon", datashuffle = False, scaleinputs = False)


    #### prediction of r at known data points 'x'

    ## get r values from the noise emulator
    x = GD.training.inputs
    x1 = __emuc.Data(x, None, GN.basis, GN.par, GN.beliefs, GN.K)
    p1 = __emuc.Posterior(x1, GN.training, GN.par, GN.beliefs, GN.K, predict=True)
    GN_mean, GN_var = p1.mean, p1.var
    r = np.exp( GN_mean + np.diag(GN_var)/2.0 )

    ## set r values in the data emulator
    GD.training.set_r(r)
    GD.training.make_A(s2 = GD.par.sigma**2 , predict=True)

    #### prediction of r at new data points 'X'

    ## get R values from the noise emulator
    X = X_inputs
    x2 = __emuc.Data(X, None, GN.basis, GN.par, GN.beliefs, GN.K)
    p2 = __emuc.Posterior(x2, GN.training, GN.par, GN.beliefs, GN.K, predict=True)
    GN_mean, GN_var = p2.mean, p2.var
    R = np.exp( GN_mean + np.diag(GN_var)/2.0 ) ## mean of noise prediction
    ## set R values for new points (X) Data object
    xp = __emuc.Data(X, None, GD.basis, GD.par, GD.beliefs, GD.K)
    xp.set_r(R)
    xp.make_A(s2 = GD.par.sigma**2 , predict=True)
    post = __emuc.Posterior(xp, GD.training, GD.par, GD.beliefs, GD.K, predict=True)
    GD_mean = post.mean ## mean of mean prediction
    
    ## return the predictive mean for data and noise
    return GD_mean, R
