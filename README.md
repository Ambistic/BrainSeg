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

The directory containing the input data must generally be specified 
using `--data`. The directory must contain a `slides` folder and a
`annotations` folder. The annotation must be an image of png format.

The ratio of the annotation dimensions must be equal to the ratio of the side dimensions, 
but the size in itself may be different.


# Full process

0) Have a folder of slides from an individual, named `SLIDE_FOLDER`
1) Use PlotFast to trace the contour and export the file as svg 
using the Ctrl + E shortcut. 
The svg files are located in the `SVG_FOLDER`. The svg stem name must be
similar to the one of the slide + the "seg" suffix.
2) The svg files must be preprocessed and converted to png in order to form
a proper mask dataset in a folder `FULL_MASK_FOLDER`. The `full_prepro_svg.py`
command is used to generate that folder.
3) These png files must be manually filled using paint on a Windows computer.
4) [Optional] After, if not all masks are filled,
the filled masks are moved to a `FILLED_MASK_FOLDER`.
5) Use the command `generate_image_dataset.py` that uses the `SLIDE_FOLDER`,
the `FILLED_MASK_FOLDER` and generates a `CURATION_DATASET_FOLDER`.
6) Use the streamlit app using `poetry run streamlit run main_streamlit.py`
Click on 001 to exclude the image and higher scores according to the
mask quality
7) Train a model using a notebook (a script is waiting to be done)
8) Use `apply.sh` to apply a model on a whole slide, this will generate
a png file
9) Use `convert_png_svg.py` to convert png back to svg. You can also use
another svg as template, predictions will be added to it (in a new file).