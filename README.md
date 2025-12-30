# (D)CNN implementation as in Long+ 2026

These files are all that is necessary to create your own custom (D)CNNs to replicate the results shown in Long+ 2026 or fine-tune your own models for your own research tasks. 

## Usage
To train a model from scratch, start a `julia` session in the same folder as which you have cloned this repository and simply type `include("ensemble.jl")` followed by `main()`. This will use the default settings to generate synthetic data and then train 
an ensemble of 20 (by default) 1D models on the generated 1D synthetic dataset. See the function `main` in `ensemble.jl` for more details on what can be fine-tuned (you can load pre-existing datasets and models, do 2D, control how the fake datasets are generated, etc.).

If you don't want to train your own models and would just like to load a sample model from the paper to make further predictions with, you can do so with syntax like: 
```julia
include("ensemble.jl")
m = load_CNN_ensemble((1001,25,1,2^6),fname="2DModelSamples/normal.jld2",nDim=2,maxChannels=32) #first arg is expected input shape (nTime, nVelocityBins, nStartChannels, nBatch)
X = zeros(1001,25,1,1) #generate some fake data to input to the model -- ideally you would replace this with lightcurve data you wanted to predict instead of just zeros
# note the shape here mostly matches the shape in loading the model above, except that nBatch does not need to be the same as specified when creating/loading the model, that's just the maximum number of samples it will iterate through at once. 
pred = m(X) #generate the prediction from the model
```
Similarly if you wanted to load i.e. the sample 1D model trained on missing data you would write: `m = load_CNN_ensemble((1001,1,2^6),fname="1DModelSamples/missing.jld2",nDim=1,maxChannels=32)`. 
Sample 1D and 2D models are provided in their respective directories. They are further detailed in the paper, but in summary all of the 1D models were trained on lightcurves with 1001 time samples (i.e. daily observational cadence for 1001 days), 
with the "missing" models trained on data with random observational gaps totaling 10% of those 1001 samples. The 2D "normal" model was similarly trained with the additional information from 25 velocity channel bins. The 2D "transfer" 
model was trained on much "worse" data extracted from a simulated observational campaign with just $\sim$ 100 lightcurve observations that were interpolated to match the 1001 samples in the other models, with 10% of those interpolated data points dropped in analogy to the 1D "missing" case. Finally, the 2D "highResTransfer" 
was trained in the same way as the "transfer" model but with 50 velocity channels (higher resolution) instead of 25. Just a single model from each ensemble is provided due to storage constraints, but if you would like access to more members of the 
ensemble for your work please feel free to email me and I will send them to you. Similarly the training/validation sets used in training these models are too large to host here, but are available on request through email or you can easily generate your 
own that should be statistically similar to those used in the paper with the code in this repository.

## Prerequisites
Assumes you have installed the following `julia` packages: CUDA, cuDNN, JLD2, Plots, Flux, ChainRulesCore, Random, Term, UnicodePlots, Distributions, DSP, FFTW, Interpolations, Optim, BroadLineRegions, Printf
By default also assumes you will be training/running the models on a system with access to an NVIDIA GPU. I believe it will work (albeit much slower) on systems without a discrete GPU
(especially if you just want to use the existing models and make new predictions) but you may have to alter a few lines of code to remove the CUDA requirements if you get errors about this. 

## Future directions
This project was a bit faster for me and I am still improving the model architecture, so things here are not "finalized" but are instead preserved as a checkpoint to document the state of the (D)CNN approach at the time of publication. In the future 
I will continue to improve this architecture and release better documented/organized code when I am fully happy with everything then, and will link to that repository from here when that is available. 
