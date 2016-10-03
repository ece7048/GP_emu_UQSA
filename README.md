# GP_emu
________

GP_emu is designed for building, training, and validating a Gaussian Process Emulator via a series of simple routines. It is supposed to encapsulate the [MUCM methodology](http://mucm.aston.ac.uk/toolkit/index.php?page=MetaHomePage.html), while also allowing the flexibility to build general emulators from combinations of kernels. In special cases, the trained emulators can be used for uncertainty and sensitivity analysis.

To install GP_emu, download the package and run the following command inside the top directory:
```
python setup.py install
```
GP_emu is written in Python, and should function in both Python 2.7+ and Python 3.


Table of Contents
=================
* [Building an Emulator](#Building an Emulator)
  * [Main Script](#Main Script)
  * [Config File](#Config File)
  * [Beliefs File](#Beliefs File)
  * [Create files automatically](#Create files automatically)
* [Design Input Data](#Design Input Data)
* [Uncertainty and Sensitivity Analysis](#Uncertainty and Sensitivity Analysis)
* [Examples](#Examples)
  * [Simple toy simulator](#Simple toy simulator)
  * [Reconstruct emulator](#Reconstruct emulator)
  * [Sensitivity examples](#Sensitivity examples)

<a name="Building an Emulator"/>
## Building an Emulator
GP_emu uses the [methodology of MUCM](http://mucm.aston.ac.uk/toolkit/index.php?page=ThreadCoreGP.html) for building, training, annd validating an emulator.

The user should create a project directory (separate from the GP_emu download), and within it place a configuration file, a beliefs file, and an emulator script containing GP_emu routines (the directory and these files can be created automatically - see [Create files automatically](#Create files automatically)). Separate inputs and outputs files should also be placed in this new directory.

The idea is to specify beliefs (e.g. form of mean function, values of hyperparameters) in the beliefs file, specify filenames and fitting options in the configuration file, and call GP_emu routines with a script. The main script can be easily editted to specifiy using different configuration files and beliefs files, allowing the user's focus to be entirely on fitting the emulator.

NOTE: emulators should be trained using a design data set such as an optimized latin hypercude design. See the section [Design Input Data](#Design Input Data).

<a name="Main Script"/>
### Main Script
This script runs a series of functions in GP_emu which automatically perform the main tasks outlined above. This allows flexibility for the user to create several different scripts for trying to fit an emulator to their data.

```
import gp_emu as g

#### configuration file
conf = g.config("toy-sim_config")

#### setup emulator
emul = g.setup(conf)

#### train emulator and run validation diagnostics
g.training_loop(emul, conf)

#### build final version of emulator
g.final_build(emul, conf)

#### plot full prediction, "mean" or "var"
g.plot(emul, [0,1],[2],[0.65], "mean")
```

The configuration file is explained later.

#### setup
The setup function needs to be supplied with the configuation object, and can optionally be supplied with two extra options e.g.:
```
emul = g.setup(conf, datashuffle=True, scaleinputs=True)
```
datashuffle=False (default True) prevents the random shuffling of the inputs and outputs (which may be useful when the data is a timeseries or when the user wishes to work on the same subset of data).

scaleinputs=False (default True) prevents each input dimension from being scaled (the input values are not adjusted at all).

#### training and validation
The functions ```training_loop()``` and ```final_build()``` are very similar, but differ:

* ```training_loop()``` trains the emulator on the current training data and validates against the current set of validation data. Prompts for retraining on new validation sets (after including the last validation set in the training set) will be made until there are no more validation sets left.

* ```final_build()``` will include the most recently used validation data set in the training set, and rebuild the emulator with this new larger training data set. No diagnostics are performed.

* An optional argument ```auto``` can be added to both ```training_loop()``` and ```final_build()``` to toggle (or explicitly set) whether the subsequent training runs and the final build will proceed automatically e.g. ```g.training_loop(emul, conf, True)``` or ```g.training_loop(emul, conf, auto=True)``` or ```g.training_loop(emul, conf, auto=False)``` etc. The default value for auto, if it is absent, is True.

* An optional argument ```message``` (default False) can be added to both ```training_loop()``` and ```final_build()``` to toggle on/off extra fitting messages from being printed (this may help identify problems with fitting an emulator).


#### Plotting
The full prediction (posterior distribution), either the mean or the variance, is displayed as a plot. Plots can be 1D (scatter plot) or 2D (colour map). For a 2D plot:

* the first list is input dimensions for (x,y) of the plot

* the second list is input dimensions to set constant values

* the third list is these constant values

If the first list contains only 1 number, then a 1D plot will be created instead (make sure the other two lists specify all other inputs and their respective constant values).

<a name="Config File"/>
### Config File
The configuration file does two things:

1. Specifies the beliefs file and data files

  * the beliefs file is explained in detail below

  * the inputs file is rows of whitespaces-separated input points, each column in the row corresponding to a different input dimensions

  * the output file is rows of output points; only one dimensional output may be specified, so each row should be a single value

2. Specifies how the emulator is trained on the data; see below

```
beliefs toy-sim_beliefs
inputs toy-sim_input
outputs toy-sim_output
tv_config 10 0 2
delta_bounds [ ]
sigma_bounds [ ]
tries 1
constraints T
stochastic T
constraints_type bounds
```
#### tv_config
Specifies how the data is divided up into training and validation sets. The data is randomly shuffled before being divided into sets (an option to turn this off may be introduced later for the purposes of training on time-series).

1. first value -- __10__ 0 2 -- how many sets to divide data into (determines size of validation set)

2. second value -- 10 __0__ 2 -- which V-set to initially validate against (currently, this should usually be set to zero; this mostly redundant feature is here for future implementation of fitting time-series data).

3. third value -- 10 0 __2__ -- number of validation sets (determines number of training points)

| tv_config | data points | T points | V-set size | V sets  |
| ----------| ------------| -------- | ---------- | ------- |
| 10 0 2    | 200         | 160      | 20         | 2       |
| 4 0 1     | 100         | 75       | 25         | 1       |
| 10 0 1    | 100         | 90       | 10         | 1       |

#### delta_bounds and sigma_bounds
Sets bounds on the hyperparameters while fitting the emulator. These bounds will only be used if constraints are specified True i.e. if
```
constraints T
```

Leaving delta_bounds and sigma_bounds 'empty'
```
delta_bounds [ ]
sigma_bounds [ ]
```
automatically constructs reasonable bounds on delta and sigma, though these might not be appropriate for the problem at hand (in which case, set constraints to false)
```
constraints F
```

To explicitly set bounds, a list of lists must be constructed, the inner lists specifying the lower and upper range on each hyperparameter, with the inner lists in the order that the hyperparameters are effectively defined due to the kernel definition.

##### delta bounds

| input dimension | kernel | delta_bounds |
| --------------- | ------ | ------------ |
| 3 | __gaussian__ + gaussian | [ __[0.0,1.0] , [0.1,0.9], [0.2,0.8]__, [0.0,1.0] , [0.1,0.9], [0.2,0.8] ] |
| 3  | gaussian + noise | [ [0.0,1.0] , [0.1,0.9], [0.2,0.8] ]  |
| 2  | 2_delta_per_dim + __gaussian__ | [ [0.0,1.0] , [0.1,0.9], _[0.0,1.0] , [0.1,0.9]_ , __[0.0,1.0] , [0.1,0.9]__ ] |

For 2_delta_per_dim there are two delta for each input dimension, so the list requires the first delta for each dimension to be specified first, followed by the second delta for each dimension i.e.
```
[ [d1(0) range] , [d1(1) range] , [d2(0) range] , [d2(1) range] ]
```

##### sigma bounds

Sigma_bounds works in the same way as delta_bounds, but is simpler since there is one sigma per kernel:

| input dimension | kernel     | sigma_bounds |
| --------------- | ---------- | ------------ |
| 2  | __gaussian__ + gaussian | [ [10.0,70.0] , [10.0,70.0] ] |
| 3  | gaussian + noise        | [ [10.0,70.0] , [0.001,0.25] ] |
| 1  | arbitrary_kernel        | [ [10.0,70.0] ] |


#### fitting options

* __tries__ : is how many times (interger) to try to fit the emulator for each training run
e.g. ``` tries 5 ```

* __constraints__ : is whether to use constraints: must be either true ```constraints T``` or false ```constraints F```

* __stochastic__ : is whether to use a stochastic 'global' optimiser ```stochastic T``` or a gradient optimser ```stochastic F```. The stohcastic optimser is slower but for well defined fits usually allows fewer tries, whereas the gradient optimser is faster but requires more tries to assure the optimum fit is found

* __constraints_type__ : can be ```constraints_type bounds``` (use the specified delta_bounds and sigma_bounds), ```constraints_type noise``` (fix the noise; only works if the last kernel is noise), or ```constraints_type standard``` (standard constraints are set to keep delta above a very small value, for numerical stability - any option that isn't bounds or noise will set constraints to standard).

<a name="Beliefs File"/>
### Beliefs File
The beliefs file specifies beliefs about the data, namely which input dimensions are active, what the mean function is believed to be, and the initial beliefs about the hyperparameter values.
```
active all
output 0
basis_str 1.0 x
basis_inf NA 0
beta 1.0 1.0
fix_mean F
kernel gaussian() noise()
delta [ ]
sigma [ ]
```

#### choosing inputs and outputs
The input dimensions to be used in building the emulator should be specified as by ```active```. If all input dimensions are to be used, then use ```active all```. If only the 0th and 2nd input dimension are to be used (indexing starts from 0), then use ```active 0 2```. (For using all input dimnesions, a list of all dimension indices can be provided instead of 'all').

The output dimension for which the emulator will be built should be specified using ```output```. Only a single index should be given, since GP_emu only works with 1 output. To build an emulator for the 2nd output dimension, use ```output 2```.

#### the mean function
This must be specified via __basis_str__ and __basis_inf__ which together define the form of the mean function. __basis_str__ defines the functions making up the mean function, and __basis_inf__ defines which input dimension those functions are for. __beta__ defines the values of the mean function hyperparameters

For mean function m(__x__) = b0
```
basis_str 1.0
basis_inf NA
beta 1.0
```

For mean function m(__x__) = 0
```
basis_str 0.0
basis_inf NA
beta 1.0
```

For mean function m(__x__) = b0 + b0x0 + b2x2
```
basis_str 1.0 x x
basis_inf NA 0 2
beta 1.0 1.0 1.0
```

For mean function m(__x__) = b0 + b0x0 + b1x1^2 + b2x2^3
```
basis_str 1.0 x   x**2 x**3
basis_inf NA  0   1    2
beta      1.0 2.0 1.1  1.6
```
In this last example, spaces have been inserted for readability.

Bear in mind that the initial values of beta, while needing to be set, do not affect the emulator fitting. However, for consistency with the belief files produced after fitting the data, which may be used to reconstruct the emulator for other purposes or may simply be used to store the fit parameters, the beta hyperparameters must be set in the initial belief file. They can all be set to 1.0, for simplicity.

The __fix_mean__ option simply allows for the mean to remain fixed at its initial specifications in the belief file. It must be ```fix_mean T``` or ```fix_mean F```.


#### Kernels
The currently available kernels are

| kernel   | class      | description |
| -------- | -----------| ----------- |
| gaussian | gaussian() | gaussian kernel |
| noise    | noise()    | additive uncorrelated noise |
| test     | test()     | (useless) example showing two length scale hyperparameters (delta) per input dimension |

More kernels can be built easily, and the following will be available in the near future: Matern, Rational Quadratic.

Kernels can be added togeter to create new kernels e.g. ```gaussian() + noise()```, which is implemented via operator overloading. To specify a list of kernels to be added together, list them in the beliefs file, separated by whitespace:

```
kernel gaussian() noise()
```

Other kernel combination operations e.g. multiplication could also be implemented, but is not priority.

Nuggets can be included in the gaussian kernel e.g. for nugget = 0.001 use ```gaussian(0.001)``` (note there are _no whitespaces_ within this term).


#### the kernel hyperparameters
The kernel hyperparameters will be automatically constructed if the lists are left empty i.e.
```
delta [ ]
sigma [ ]
```
which is recommended as the initial values do not affect how the emulator is fit. However, for consistency with the beliefs file produced after training (and to explain that file), the kernel hyperparameter beliefs can be specified below. However, the easiest way to construct these lists is to run the program with empty lists, and then copy and paste the lists from the updated beliefs files into the initial belief file (specified in the configuration file).

##### delta
The following shows how to construct the lists piece by piece.

1. a list for each kernel being used e.g. K = gaussian() + noise() we need ```delta [ [ ] , [ ] ]```

2. within each kernel list, d lists, where d is the number of hyperparameters per dimension
 * if there is one delta per input dimension for K = one_delta_per_dim() we need ```[ [ [ ] ] ]```
 * if there are two delta per input dimenstion for K = two_delta_per_dim() we need  ```[ [ [ ] , [ ] ] ]``` i.e. within the kernel list we have two lists in which to specify the delta for the first input dimension and the second input dimension
 * so for K = two_delta_per_dim() + one_delta_per_dim() we need ```[  [ [ ],[ ] ]  ,  [ [ ] ]  ]```

Within these inner most lists, the n values of delta (n is the number of dimensions) should be specified.

e.g. K = one_delta_per_dim() in 1D we need ```[ [ [1.0] ] ]```

e.g. K = one_delta_per_dim() in 2D we need ```[ [ [1.0, 1.0] ] ]```

e.g. K = two_delta_per_dim() in 1D we need  ```[ [ [1.0] , [1.0] ] ]```

e.g. K = two_delta_per_dim() in 2D we need  ```[ [ [1.0,1.0] , [1.0,1.0] ] ]```

e.g. K = gaussian() + gaussian() in 1D we need ```[ [ [1.0] ] , [ [1.0] ] ]```

e.g. K = gaussian() + gaussian() in 2D we need ```[ [ [1.0,1.0] ] , [ [1.0, 1.0] ] ]```

e.g. K = gaussian() + noise() in 2D we need
```
delta [ [ [0.2506, 0.1792] ] , [ ] ]
```
_If a kernel has no delta values, such as the noise kernel, then its list should be left empty._

##### sigma
Sigma is simpler, as there is one per kernel:

e.g. K = gaussian() in 1D we need ``` sigma [ [0.6344] ]```

e.g. K = gaussian() in 2D we need ``` sigma [ [0.6344] ]```

e.g. K = gaussian() + noise() in 1D we need ``` sigma [ [0.6344] , [0.0010] ]```

e.g. K = gaussian() + noise() in 2D we need ``` sigma [ [0.6344] , [0.0010] ]```


<a name="Create files automatically"/>
### Create files automatically
A routine ```create_emulator_files()``` is provided to create a directory (inside of the current directory) containing default belief, config, and main script files. This is to allow the user to easily set up different emulators (an example application is when building a separate emulator for each output of a simulator). User editting of the generated files is generally necessary.

It is simplest to run this function from an interactive python session as follows:
```
>>> import gp_emu as g
>>> g.create_emulator_files()
```
The function will then prompt the user for input.

<a name="Design Input Data"/>
## Design Input Data
See the following page for [MUCM's discussion on data design](http://mucm.aston.ac.uk/toolkit/index.php?page=AltCoreDesign.html)

To import this subpackage use something like this
```
import gp_emu.design_inputs as d
```
Currently, only an optimised Latin Hypercube design is included.

Scripts should be written to configure the design and run the design function e.g.

```
import gp_emu.design_inputs as d

#### configuration of design inputs
dim = 2
n = 60
N = 200
minmax = [ [0.0,1.0] , [0.0,1.0] ]
filename = "toy-sim_input"

#### call function to generate input file
d.optLatinHyperCube(dim, n, N, minmax, filename)
```
The design input points, output to _filename_, are suitable for reading by GP_emu: each line (row) represents one input point of _dim_ dimensions.


<a name="Uncertainty and Sensitivity Analysis"/>
##Uncertainty and Sensitivity Analysis
See the following pages for MUCM's discussions on [uncertainty quantification](http://mucm.aston.ac.uk/toolkit/index.php?page=DiscUncertaintyAnalysis.html) and [sensitivity analysis](http://mucm.aston.ac.uk/toolkit/index.php?page=ThreadTopicSensitivityAnalysis.html).

Having constructed an emulator with GP_emu, the subpackage GP_emu can be used to perform sensitivity analysis. The routines only work for an emulator with a Gaussian kernel (with or without nugget) and linear mean function, and the inputs (of the model for which the emulator is built) are assumed to be independant and normally distributed with mean m and variance v (support for emulators with a generalised mean function and correlated inputs may be added in the future).

### Setup

Include the sensitivity subpackage as follows:
```
import gp_emu.sensitivity as s
```

A distribution for the inputs must be defined by a mean m and variance v for each input. These means and variances should be stored as a list e.g. for an emulator with three inputs with mean 0.50 and variance 0.02 for each input:

```
m = [0.50, 0.50, 0.50]
v = [0.02, 0.02, 0.02]
```

These lists and the emulator "emul" must then be passed to the a setup function which returns a sensitivity class instance:

```
sens = s.setup(emul, m, v)
```

### Routines

#### Uncertainty

To perform uncertainty analysis to calculate, with respect to the emulator, the expection of the expection, the expection of the variance, and the variance of the expectation, use:
```
sens.uncertainty()
```

#### Sensitivity

To calculate sensitivity indices for each input, use:
```
sens.sensitivity()
```

#### Main Effect
To calculate and plot the main effects of each input, and optionally plot (plot default=False) them, use:
```
sens.main_effect(plot=True)
```
The number of points in the (scaled) input ranges 0.0 to 1.0 for each input to use can optionally be specified too (the default is 100):
```
sens.main_effect(plot=True, points = 200)
```
Extra optional arguments for the plot can also be chosen for the key, labels, and to control the plot scale (useful for adjusting the exact placement of the key):
```
sens.main_effect(plot=True, customKey=['Na','K'], customLabels=['Model Inputs','Main Effect for dV/dt'], plotShrink=0.9)
```
An optional argument for the subset of inputs to be calculated/plotted can be provide (default is all inputs) e.g. to plot only input 0 and input 2:
```
sens.main_effect(plot=False, w=[0,2])
```

#### Interaction Effect

The interaction effect between two inputs {i,j} can be calculated and plotted with:
```
sens.interaction_effect(i, j)
```
Optional arguments can be supplied to specify the number of points used in each dimension (default=25, so the 2D plot will consist of 25*25 = 625 points) and labels for the x and y axes.
```
sens.interaction_effect(i, j, points = 25, customLabels=["input i", "input j"])
```

#### Total Effect Variance

The total effect variance for each input can be calculated with:
```
sens.totaleffectvariance()
```

#### Save results to file

To save calculated sensitivity results to file, use the to_file function:
```
sens.to_file("test_sense_file")
```


### Plot a sensitivity table
To plot a sensitivity table of the sensitivities divided by the expection of the variance, use
```
s.sense_table([sens,], [], [])
```
where the first argument is a list containing sensitivity objects, the second list can be filled with labels for the table columns (inputs), and the third list can be filled with labels for the table rows (outputs).

By looping over different emulators (built to emulate a different outputs) and building up a list of sensitivites, it is possible to call sense_table with a list of the results of the sensitivity calculations for each emulator. Calling sense_table will then results in a table with columns of inputs and rows of outputs (each row corresponding to an emulator built for a different output).

```
sense_list = [ ]
for i in range(num_emulators):

    ... build/load emulator etc. ...

    ... call the uncertainty and senstiivty routines etc...

    sense_list.append(sens)
# end of loop

s.sense_table(sense_list, [], [])
```
An optional integer argument to sense_table is possible ```s.sense_table(sense_list, [], [], 4)``` which changes the height of the rows in the resulting table. The user might want to call sense_table several times with different integers until the table looks right, and then use just the one function call with that number.


<a name="Examples"/>
## Examples

<a name="Simple toy simulator"/>
### Simple toy simulator
To run a simple example, do
```
cd examples/toy-sim/
python emulator.py
```
The script emulator.py will attempt to build an emulator from the data found in toy-sim_input and toy-sim_output:
* toy-sim_input contains 2 dimensional inputs generated from an optimised latin hypercube design
* toy-sim_output contains the 1 dimensional output generated from a simulation which takes 2 inputs

The script toy-sim.py is the 'toy simulation': it is simply a deterministic function performing some operations on several numbers and returning a single number. This script can be run with
```
python toy-sim.py toy-sim_input
```
or, for additive random noise from a normal distribution, with
```
python toy-sim.py toy-sim_input 0.25
```
where 0.25 is the amplitude multiplying the noise in this example.

The user must specify the input file on the command line, so other input files can be used. Using the design_inputs subpackage, other input files (with, say, more or less points, and more or less dimensions) can be generated in order to run this example (this example will run with up to 3 dimensional input).

The underlying function that the emulator attempts to reconstruct can easily be changed in toy-sim.py, and functions taking higher dimensional input (4 inputs, 5 inputs etc.) can be easily added. The script emulator.py is configured to plot only the first two dimensions, while holding other dimensions constant, but this can be easily modified.

If adding noise to the toy simulation, then ```kernel gaussian() noise()``` could be specified in the belief file.

<a name="Reconstruct emulator"/>
### Reconstruct emulator from saved files

When building an emulator, several files are saved at each step: an updated beliefs file and the inputs and outputs used in the construction of the emulator. The emulator can be rebuilt from these files without requiring another training run or build, since all the information is specified in these files. A minimal script would be:

```
import gp_emu as g

conf = g.config("toy-sim_config_reconst")
emul = g.setup(conf)
g.plot(emul, [0,1],[2],[0.3], "mean")
```
where "toy-sim_config_reconst" contains the names of files generated from the final build after the second training step:
```
beliefs toy-sim_beliefs-2f
inputs toy-sim_input-o0-2f
outputs toy-sim_output-o0-2f
```

Be careful to specify the output correctly in the new beliefs file - if using output 1 to specify using the second column in the original outputs file, then the newer file 'toy-sim_output-o1-2f' (for example) will contain only a single column of outputs (the second column from the original output file). Thus the new beliefs file should specifiy that output 0 should be used, since we wish to use the first (and only) column of outputs from the updated output file.
The same case applies to updated input files when only a subset of the input dimensions have been used (by specifying 'active' in the beliefs file to be a non-empty list of integers). If inputs [0,2] out of original inputs [0,1,2] have been specified as active, then the updated inputs files will contain only these inputs, which will now be referred to by indices [0,1] since these are the columns in the updated inputs file.

Be especially careful with the tv_config option: if you wish to reconstruct the emulator from all inputs (without calling the training_loop or final_build functions) then the last value of tv_config (which specifies how many validation sets to use) should be set to 0. This way, the emulator that is rebuilt is exactly the same as the emulator that was rebuilt when the updated beliefs, inputs and output files were saved.

<a name="Sensitivity Examples"/>
### Sensitivity examples

#### surfebm
This example demonstrates building an emulator and performing sensitivity analysis as in the example here: http://mucm.aston.ac.uk/MUCM/MUCMToolkit/index.php?page=ExamCoreGP2Dim.html

#### multiple outputs
This example demonstrates building an emulators for simulations with multiple outputs. A separate emulator is built for each output, and by looping over different emulators, it is possible to build a sensitivity table showing how all the outputs depend on the inputs. Note that we need multiple config files and belief files specified, since we need to indicate the output we are building and emulator for in the belief file, and need to indicate the belief file we're using in the config file.