# BrainSeg

Project of automatic brain segmentation using Deep Learning.

## Installation

The installation can be done using the `install.sh` command.
It will require poetry, and install the full environment.

## Process

The segmentation is a two-step process : training and application.

### Training

The data must be first preprocessed

### Application

Run the command apply.sh --data `data_dir`

## Data

The directory containing the input data must generally be specify 
using `--data`. The directory must contain a `slides` folder and a
`annotations` folder. The annotation must be an image of png format.

The ratio of the annotation dimensions must be equal to the ratio of the side dimensions, but the size in itself may be different.
