from __future__ import print_function
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
#from scipy.optimize import differential_evolution
import time

import sys
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    #print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    print('\x08%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r', flush=True)
    #sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    #sys.stdout.flush()

    # Print New Line on Complete
    if iteration == total: 
        print()

from scipy.optimize import check_grad

## use '@timeit' to decorate a function for timing
def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        for r in range(100): # calls function 100 times
            result = f(*args, **kw)
        te = time.time()
        print('func: %r took: %2.4f sec' % (f.__name__, te-ts) )
        return result
    return timed

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

### for optimising the hyperparameters
class Optimize:
    def __init__(self, data, basis, par, beliefs, config):
        self.data = data
        self.basis = basis
        self.par = par
        self.beliefs = beliefs
        self.config = config

        ## for message printing - off by default
        self.print_message = False

        print("\n*** Optimization options ***")

        # if bounds are empty then construct them automatically
        d_bounds_t = []
        n_bounds_t = []
        s_bounds_t = []
 
        if config.delta_bounds == []:
            print("Data-based bounds for delta:")
            # loop over the dimensions of the inputs for delta
            for i in range(0, self.data.inputs[0].size):
                data_range = np.amax(self.data.inputs[:,i])\
                           - np.amin(self.data.inputs[:,i])
                d_bounds_t.append([0.01,data_range])
                print("    delta" , i , '[{:04.4f} , {:04.4f}]'.format(d_bounds_t[i][0] , d_bounds_t[i][1]) )
        else:
            print("User provided bounds for delta:")

            if len(config.delta_bounds) != self.data.inputs[0].size:
                print("ERROR: Wrong number of delta_bounds specified, exiting.")
                exit()

            for i in range(0, self.data.inputs[0].size):
                if config.delta_bounds[i] == []:
                    data_range = np.amax(self.data.inputs[:,i])\
                               - np.amin(self.data.inputs[:,i])
                    d_bounds_t.append([0.01,data_range])
                    print("    delta" , i , '[{:04.4f} , {:04.4f}]'.format(d_bounds_t[i][0] , d_bounds_t[i][1]), "(data)")
                else:
                    d_bounds_t.append(config.delta_bounds[i])
                    print("    delta" , i , '[{:04.4f} , {:04.4f}]'.format(d_bounds_t[i][0] , d_bounds_t[i][1]), "(user)")

            #for i in range(0, self.data.inputs[0].size):
            #    print("    delta" , i , d_bounds_t[i])

        if config.nugget_bounds == []:
            print("Data-based bounds for nugget:")
            # use small range for nugget
            data_range = np.sqrt( np.amax(self.data.outputs)\
                       - np.amin(self.data.outputs) )
            n_bounds_t.append([0.0001,0.01])
            print("    nugget " , '[{:04.4f} , {:04.4f}]'.format(n_bounds_t[0][0] , n_bounds_t[0][1]) )
        else:
            print("User provided bounds for nugget:")
            n_bounds_t = config.nugget_bounds
            print("    nugget " , '[{:04.4f} , {:04.4f}]'.format(n_bounds_t[0][0] , n_bounds_t[0][1]) )

        if config.sigma_bounds == []:
            print("Data-based bounds for sigma:")
            # use output range for sigma
            data_range = np.sqrt( np.amax(self.data.outputs)\
                       - np.amin(self.data.outputs) )
            s_bounds_t.append([0.01,data_range])
            print("    sigma  " , '[{:04.4f} , {:04.4f}]'.format(s_bounds_t[0][0] , s_bounds_t[0][1]) )
        else:
            print("User provided bounds for sigma:")
            s_bounds_t = config.sigma_bounds
            print("    sigma  " , '[{:04.4f} , {:04.4f}]'.format(s_bounds_t[0][0] , s_bounds_t[0][1]) )

        ## different bounds depending on scenario
        if self.beliefs.fix_nugget == 'F':
            if self.beliefs.mucm == 'T':
                config.bounds = tuple(d_bounds_t + n_bounds_t)
            else:
                config.bounds = tuple(d_bounds_t + n_bounds_t + s_bounds_t)
        else:
            if self.beliefs.mucm == 'T':
                config.bounds = tuple(d_bounds_t)
            else:
                config.bounds = tuple(d_bounds_t + s_bounds_t)
        #print("bounds:" , config.bounds)

        # set up type of bounds
        if config.constraints == "bounds":
            self.bounds_constraint(config.bounds)
        else:
            self.standard_constraint(config.bounds)

        
    ## tries to keep deltas above a small value
    def standard_constraint(self, bounds):
        print("Setting up standard constraint")
        self.cons = []

        d_size = self.data.K.d.size
        for i in range(0, d_size):
            self.cons.append([self.data.K.transform(0.01),None])
         
        if self.beliefs.fix_nugget == 'F':
            self.cons.append([None,None])
        if self.beliefs.mucm == 'F':
            self.cons.append([None,None])

        return 


    ## tries to keep within the specified bounds
    def bounds_constraint(self, bounds):
        print("Setting up bounds constraint")
        self.cons = []

        x_size = self.data.K.d.size
        if self.beliefs.fix_nugget == 'F':
            if self.beliefs.mucm == 'T':
                x_size = x_size + 1
            else:
                x_size = x_size + 2
        else:
            if self.beliefs.mucm == 'T':
                x_size = x_size
            else:
                x_size = x_size + 1

        for i in range(0, x_size):
            lower, upper = bounds[i]
            self.cons.append([self.data.K.transform(lower),\
                              self.data.K.transform(upper)])

        return


    def llh_optimize(self, print_message=False):

        numguesses = self.config.tries
        bounds = self.config.bounds
        self.print_message = print_message

        print("Optimising hyperparameters...")

        ## transform the provided bounds
        bounds = self.data.K.transform(bounds)
 
        ## actual function containing the optimizer calls
        self.optimal(numguesses, bounds)

        print("best hyperparameters: ")
        self.data.K.print_kernel()
        print("sigma:" , np.round(self.par.sigma,decimals=6))

        if self.beliefs.fix_nugget == 'F':
            noisesig = np.sqrt(self.par.sigma**2 * (self.par.nugget)/(1.0-self.par.nugget))
            print("'noise sigma' estimate from nugget:" , noisesig)
        
        self.optimalbeta()
        print("best beta: " , self.par.beta)

   
    def optimal(self, numguesses, bounds):
        first_try = True
        best_min = 10000000.0

        ## params - number of paramaters that need fitting
        params = self.data.K.d.size
        if self.beliefs.fix_nugget == 'F':
            if self.beliefs.mucm == 'T': params = params + 1
            else: params = params + 2
        else:
            if self.beliefs.mucm == 'T': params = params
            else: params = params + 1
        
        ## construct list of guesses from bounds
        guessgrid = np.zeros([params, numguesses])
        print("Calculating initial guesses from bounds")
        for R in range(0, params):
            BL, BU = bounds[R][0], bounds[R][1]
            guessgrid[R,:] = BL+(BU-BL)*np.random.random_sample(numguesses)

        ## print information about which parameters we're fitting
        if self.beliefs.fix_nugget == 'F': print("Training nugget on data")

        ## print information about which method we're using
        if self.beliefs.mucm == 'T': print("Using MUCM method for sigma")

        ## set which LLH expression to use
        llh = self.loglikelihood_mucm if self.beliefs.mucm == 'T' else self.loglikelihood_gp4ml

        ## tell user which fitting method is being used
        if self.config.constraints != "none": print("Using L-BFGS-G method (with constraints)...")
        else: print("Using L-BFGS-G method (no constraints)...")

        ## try each x-guess (start value for optimisation)
        myprint = False
        printProgressBar(0, numguesses, prefix = 'Progress:', suffix = '', length = 25)
        for C in range(0,numguesses):
            x_guess = list(guessgrid[:,C])
            #print(C, x_guess, numguesses)
            #input()
            print("\nInitial guess:", self.data.K.untransform(np.array(x_guess)))

            nonPSDfail = False
            try:
                if self.config.constraints != "none":
                    res = minimize(llh, x_guess, method = 'L-BFGS-B', jac=True, bounds=self.cons)
                else:
                    res = minimize(llh, x_guess, method = 'L-BFGS-B', jac=True)

                ## for checks on if function gradient is correct
                debug_grad = False
                if debug_grad:
                    func_m = lambda x: self.loglikelihood_mucm(x, debug="func")
                    grad_m = lambda x: self.loglikelihood_mucm(x, debug="grad")
                    func_g = lambda x: self.loglikelihood_gp4ml(x, debug="func")
                    grad_g = lambda x: self.loglikelihood_gp4ml(x, debug="grad")
                    func, grad = (func_m, grad_m) if self.beliefs.mucm == 'T' else (func_g, grad_g)
                    print("  grad error initial guess:", check_grad(func, grad, x_guess))
                    print("  grad error optimized val:", check_grad(func, grad, res.x))

            except TypeError as e:
                nonPSDfail = True

            ## check that we didn't fail by having non-PSD matrix
            if nonPSDfail == False:
                if self.print_message: print(res)

                ## check more than 1 iteration was done
                nfev = res.nfev
                not_fit = True if nfev == 1 else False

                if self.print_message:
                    print(res, "\n")
                    if res.success != True:
                        print(res.message, "Not succcessful.")

                ## result of fit
                sig_str = "" 
                if self.beliefs.mucm == 'T':
                    self.sigma_analytic_mucm(self.data.K.untransform(res.x))
                    sig_str = "  sig: " + str(np.around(self.par.sigma,decimals=4))

                #if myprint == True: 
                print("  hp: ",\
                        np.around(self.data.K.untransform(res.x),decimals=4),\
                        " llh: ", -1.0*np.around(res.fun,decimals=4) , sig_str)
                ## result of fit
                HP = np.around(self.data.K.untransform(res.x),decimals=4)
                if not_fit == False and res.success == True: # if successful
                    sig_str = "" 
                    if self.beliefs.mucm == 'T':
                        self.sigma_analytic_mucm(self.data.K.untransform(res.x))
                        sig_str = "  sig: " + str(np.around(self.par.sigma,decimals=4))
                    print("  hp: ", HP, " llh: ", -1.0*np.around(res.fun,decimals=4) , sig_str)
                else: # if unsuccessful
                    if not_fit: print("  WARNING: Only 1 iteration for", HP, ", not fitted.")
                    if res.success == False: print("  WARNING: Unsuccessful termination for", HP, ", not fitted.")
                if self.print_message: print("\n")
                    
                ## set best result
                if (res.fun < best_min or first_try) and not_fit == False and res.success == True:
                    best_min, best_x = res.fun, self.data.K.untransform(res.x)
                    first_try = False

            else:
                print("Trying next guess...")

            # Update Progress Bar
            printProgressBar(C + 1, numguesses, prefix = 'Progress:', suffix = '', length = 25)


        print("********")
        if first_try == False:
            if self.beliefs.mucm == 'T':
                self.data.K.set_params(best_x)
                self.sigma_analytic_mucm(best_x)
            else:
                self.data.K.set_params(best_x[:-1])
                self.par.sigma = best_x[-1]
            self.par.delta = self.data.K.d
            self.par.nugget = self.data.K.n

            self.data.make_A(self.par.sigma**2)
            self.data.make_H()
        else:
            print("ERROR: No optimization was made. Exiting.")
            exit()



    ## the loglikelihood provided by MUCM
    def loglikelihood_mucm(self, x, debug=False):
        if debug != False: x = np.array(x)
        x = self.data.K.untransform(x)
        self.data.K.set_params(x)
        self.data.make_A()

        n, q = self.data.inputs[:,0].size, self.par.beta.size

        try:
            L = np.linalg.cholesky(self.data.A) 
            w = np.linalg.solve(L,self.data.H)
            Q = w.T.dot(w)
            K = np.linalg.cholesky(Q)
            invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs))
            invA_H = np.linalg.solve(L.T, np.linalg.solve(L,self.data.H))

            solve_K_HT = np.linalg.solve(K,self.data.H.T)
            B = np.linalg.solve(K.T, solve_K_HT.dot(invA_f))

            invA_H_dot_B = invA_H.dot(B)
            sig2 = ( 1.0/(n - q - 2.0) )*\
                   np.transpose(self.data.outputs).dot(invA_f-invA_H_dot_B)

            self.par.sigma = np.sqrt(sig2)

            logdetA = 2.0*np.sum(np.log(np.diag(L)))

            LLH = -0.5*(\
                        -(n - q) * np.log(sig2)\
                        -logdetA\
                        -np.log(np.linalg.det(Q))\
                       )

            ## calculate the gradients wrt hyperparameters
            grad_LLH = np.empty(x.size)
            
            H_dot_B = self.data.H.dot(B).T
           
            factor = (n - q) / (sig2*(n - q - 2))

            #### wrt delta
            for i in range(self.data.K.d.size):
                temp = self.data.K.grad_delta_A(self.data.inputs[:,i], i, sig2)
                invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
                sam = (invA_gradHP).dot(invA_H_dot_B)
                grad_LLH[i] = -0.5* (\
                  -np.trace(invA_gradHP) \
                  + factor * ( \
                    +(self.data.outputs.T).dot(invA_gradHP).dot(invA_f) \
                    +( - 2*self.data.outputs.T + H_dot_B ).dot(sam) \
                             ) \
                  + np.trace( np.linalg.solve(K.T, \
                                solve_K_HT.dot(invA_gradHP)).dot(invA_H) )
                                    )

            #### wrt nugget
            if x.size == self.data.K.d.size + 1: ## if x contains nugget value
                temp = self.data.K.grad_nugget_A(self.data.inputs, sig2)
                invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
                sam = (invA_gradHP).dot(invA_H_dot_B)
                grad_LLH[x.size-1] = -0.5* (\
                  -np.trace(invA_gradHP) \
                  + factor * ( \
                    +(self.data.outputs.T).dot(invA_gradHP).dot(invA_f) \
                    +( - 2*self.data.outputs.T + H_dot_B ).dot(sam) \
                             ) \
                  + np.trace( np.linalg.solve(K.T, \
                                solve_K_HT.dot(invA_gradHP)).dot(invA_H) )
                                           )

        except np.linalg.linalg.LinAlgError as e:
            print("  WARNING: Matrix not PSD for", x, ", not fitted.")
            return None

        if debug == False:
            return LLH, grad_LLH
        elif debug == "func":
            return LLH
        elif debug == "grad":
            return grad_LLH


    ## calculate sigma analytically - used for the MUCM method
    def sigma_analytic_mucm(self, x):
        ## to match my covariance matrix to the MUCM matrix 'A'
        self.data.K.set_params(x)
        self.data.make_A()

        n, q = self.data.inputs[:,0].size, self.par.beta.size

        try:
            L = np.linalg.cholesky(self.data.A) 
            w = np.linalg.solve(L,self.data.H)
            Q = w.T.dot(w)
            K = np.linalg.cholesky(Q)
            invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs))
            invA_H = np.linalg.solve(L.T, np.linalg.solve(L,self.data.H))
            B = np.linalg.solve(K.T, np.linalg.solve(K,self.data.H.T).dot(invA_f))

            sig2 = ( 1.0/(n - q - 2.0) )*\
                   np.transpose(self.data.outputs).dot(invA_f-invA_H.dot(B))

            self.par.sigma = np.sqrt(sig2)

        except np.linalg.linalg.LinAlgError as e:
            print("  WARNING: In sigma_analytic_mucm(): "
                  "Matrix not PSD for", x, ", not fitted.")
            exit()

        return


    ## the loglikelihood provided by Gaussian Processes for Machine Learning 
    def loglikelihood_gp4ml(self, x, debug=False):
        if debug != False: x = np.array(x)
        x = self.data.K.untransform(x)
        self.data.K.set_params(x[:-1]) # not including sigma in x
        self.par.sigma = x[-1]
        s2 = x[-1]**2
        self.data.make_A(s2)

        ## for now, let's just multiply A by sigma**2
        self.data.A = s2*self.data.A

        n, q = self.data.inputs[:,0].size, self.par.beta.size

        try:
            L = np.linalg.cholesky(self.data.A) 
            w = np.linalg.solve(L,self.data.H)
            Q = w.T.dot(w) # H A^-1 H
            K = np.linalg.cholesky(Q)
            invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs)) # A^-1 y
            invA_H = np.linalg.solve(L.T, np.linalg.solve(L,self.data.H)) # A^-1 H

            solve_K_HT = np.linalg.solve(K,self.data.H.T)
            B = np.linalg.solve(K.T, solve_K_HT.dot(invA_f)) # (H A^-1 H)^-1 H A^-1 y

            logdetA = 2.0*np.sum(np.log(np.diag(L)))

            invA_H_dot_B = invA_H.dot(B) # A^-1 H (H A^-1 H)^-1 H A^-1 y
            longexp = ( self.data.outputs.T ).dot( invA_f - invA_H_dot_B )
            # y A^-1 y - y A^-1 H (H A^-1 H)^-1 H A^-1 y )

            LLH = -0.5*\
              (-longexp - logdetA - np.log(linalg.det(Q))\
               -(n - q)*np.log(2.0*np.pi))

            ## calculate the gradients wrt hyperparameters
            grad_LLH = np.empty(x.size)
            
            H_dot_B = self.data.H.dot(B).T
            # H (H A^-1 H)^-1 H A^-1 y
            
            #### wrt delta
            for i in range(self.data.K.d.size):
                temp = self.data.K.grad_delta_A(self.data.inputs[:,i], i, s2)
                invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
                sam = (invA_gradHP).dot(invA_H_dot_B)
                grad_LLH[i] = -0.5* (\
                  -np.trace(invA_gradHP) \
                  +(self.data.outputs.T).dot(invA_gradHP).dot(invA_f) \
                  +( - 2*self.data.outputs.T + H_dot_B ).dot(sam) \
                  + np.trace( np.linalg.solve(K.T, \
                                solve_K_HT.dot(invA_gradHP)).dot(invA_H) )
                                    )

            #### wrt nugget
            if x.size == self.data.K.d.size + 2: ## if x contains nugget value
                temp = self.data.K.grad_nugget_A(self.data.inputs, s2)
                invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
                sam = (invA_gradHP).dot(invA_H_dot_B)
                grad_LLH[x.size-2] = -0.5* (\
                  -np.trace(invA_gradHP) \
                  +(self.data.outputs.T).dot(invA_gradHP).dot(invA_f) \
                  +( - 2*self.data.outputs.T + H_dot_B ).dot(sam) \
                  + np.trace( np.linalg.solve(K.T, \
                                solve_K_HT.dot(invA_gradHP)).dot(invA_H) )
                                           )

            #### wrt sigma ## in gp4ml LLH sigma is always in x
            temp = self.data.A ## already X s2
            np.fill_diagonal(temp, temp.diagonal() - self.data.r) ## correct for extra variance
            invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
            sam = (invA_gradHP).dot(invA_H_dot_B)
            grad_LLH[x.size-1] = -0.5* (\
              -np.trace(invA_gradHP) \
              +(self.data.outputs.T).dot(invA_gradHP).dot(invA_f) \
              +( - 2*self.data.outputs.T + H_dot_B ).dot(sam) \
              + np.trace( np.linalg.solve(K.T, \
                            solve_K_HT.dot(invA_gradHP)).dot(invA_H) )
                                )

        except np.linalg.linalg.LinAlgError as e:
            print("  WARNING: Matrix not PSD for", x, ", not fitted.")
            return None

        if debug == False:
            return LLH, grad_LLH
        elif debug == "func":
            return LLH
        elif debug == "grad":
            return grad_LLH


    # calculates the optimal value of the mean hyperparameters
    def optimalbeta(self):
        L = np.linalg.cholesky(self.data.A) 
        w = np.linalg.solve(L,self.data.H)
        Q = w.T.dot(w)
        K = np.linalg.cholesky(Q)
        invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs))
        self.par.beta =\
          np.linalg.solve(K.T, np.linalg.solve(K,self.data.H.T).dot(invA_f))


