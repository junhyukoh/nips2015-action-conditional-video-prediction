# Introduction
This repository implements the main algorithm of the following paper ([Project website](https://sites.google.com/a/umich.edu/junhyuk-oh/action-conditional-video-prediction)):
  * Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard Lewis, Satinder Singh, **"Action-Conditional Video Prediction using Deep Networks in Atari Games**"
    _In Advances in Neural Information Processing Systems (NIPS)_, 2015.

```
@incollection{NIPS2015_5859,
author = {Oh, Junhyuk and Guo, Xiaoxiao and Lee, Honglak and Lewis, Richard L and Singh, Satinder},
booktitle = {Advances in Neural Information Processing Systems 28},
editor = {Cortes, C and Lawrence, N D and Lee, D D and Sugiyama, M and Garnett, R and Garnett, R},
pages = {2845--2853},
publisher = {Curran Associates, Inc.},
title = {{Action-Conditional Video Prediction using Deep Networks in Atari Games}},
year = {2015}
}
```

# Installation
This repository contains a modified version of Caffe and uses its python wrapper (pycaffe). <br />
Please check the following instruction to compile Caffe:
http://caffe.berkeleyvision.org/installation.html. <br />
After installing the libraries required by Caffe, you should be able to compile the code succesfully as follows:

```
cd caffe
make
make pycaffe
```

# Data structure
The data directories should be organized as follows:
```
./[game name]/train/[%04d]/[%05d].png  # training images
./[game name]/train/[%04d]/act.log     # training actions
./[game name]/test/[%04d]/[%05d].png   # testing images
./[game name]/test/[%04d]/act.log      # testing actions
./[game name]/mean.binaryproto         # mean pixel image
```
`[%04d]` and `[%05d]` correspond to `episode index` and `frame index` respectively (starting from 0). <br />
Each line of `act.log` file specifies the action index (starting from 0) chosen by the player for each time step. <br />
```
[action idx at time 0]
[action idx at time 1]
[action idx at time 2]
...
```
The mean pixel values should be computed over the entire training images and be converted to `binaryproto` using Caffe. <br />

# Training
The following scripts are provided for training:
  * `train_cnn.sh` : train a feedforward model on 1-step, 3-step, 5-step objectives.
  * `train_lstm.sh` : train a recurrent model on 1-step, 3-step, 5-step objectives.
  * `train.sh` : train any types of models with user-specified details (batch_size, pre-trained weights, etc)

The following command shows how to run training scripts:
```
cd [game name]
../train_cnn.sh [num_actions] [gpu_id]
../train_lstm.sh [num_actions] [gpu_id]
../train.sh [model_type] [result_prefix] [lr] [num_act] [...]
```

# Testing
The following scripts are provided for testing:
  * `test_cnn.sh` : shows predictions from a trained feedforward model.
  * `test_lstm.sh` : shows predictions from a trained recurrent model.
  * `test.sh` : shows predictions from a trained model with user-specified details

The following command shows how to run the testing script:
```
cd [game name]
../test_cnn.sh [weights] [num_actions] [num_step] [gpu_id]
../test_lstm.sh [weights] [num_actions] [num_step] [gpu_id]
../test.sh [model_type] [weights] [num_action] [num_input_frames] [num_step] [gpu_id] [...]
```

  * If `line 31` of `test.py` gives an error, you have to replace the default font path with a path for any fonts
```
font = ImageFont.truetype('[path for a font]', 20)
```

# Details
This repository uses `ADAM` optimization method, while `RMSProp` is used in the original paper.
We found that `ADAM` converges more quickly, and 3-step training is almost enough to get reasonable results.
