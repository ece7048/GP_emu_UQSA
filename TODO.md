# TODO

## general
* Check the diagonal correction in the gradient of loglikelihood_gp4ml function
* The LLH gradients may have errors, despite optimizing to same values... needs rechecking


## sensitivity


## noise fit
* zp_outputs is only made properly when datasize is divisible by the number of sets


## history matching
* When previously-built emulators use scaleinputs=False, the empty input_minmax list causes error in emulsetup - need a better way to save and use minmax info in beliefs file
* Need to translate NIMP points back into original units for simulation
* simplify new_inputs() and first_design() so that a user can simply pass the NIMP values; this means they can pass NIMP to their own function instead e.g. use diversipy python package
* retrieve non-imp simulation inputs which we already have
* possibly replace NIMP and NIMP_I with list of indices into test points - saves storage
* make way to combine separate waves into one data structure


## design inputs
* Update docs for optimalLatinHyperCube() function
