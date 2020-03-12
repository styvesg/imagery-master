# Generative feedback explains distinct brain activity codes for seen and mental images
## Notebooks and code
### gabor_fwrf_model_training
A notebook to train the model parameters with k-fold validation for vision and imagery components of the experiment.

Input:

- Stimuli.h5py
- Stimuli_metadata.pkl
- Voxels.h5py
- Voxels_metadata.pkl

Produces a model parameter file:

- fwrf_{subject}_{timestamp}_data.pkl

### linear_brain_model
Create a linear brain model from a set of structural prescriptions and perform vision and imagery-hypothetized inference from different area and explore its effects.

Produces:

- Figures
