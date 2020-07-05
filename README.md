# Shopee Product Detection

8th place solution with a private score of `0.84720`

## Setup
Python 3.7+

To install a virtual environment with [pipenv](https://github.com/pypa/pipenv):
```
$ pipenv install -r requirements.txt --python 3.7
```
Also, we use a custom lr scheduler from this [github](https://github.com/ildoonet/pytorch-gradual-warmup-lr):
```
$ pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
``` 
**_Optional:_** To install apex, please check [this](https://github.com/NVIDIA/apex) out

## Training

Before training/reproducing a model, you might need to customise the arguments
in `run.sh` to fit with your environment.

+ `--fp16`: Set it as `True` if you want to train a model with mixed-precision.
This requires [apex](https://github.com/NVIDIA/apex) to be installed
+ `--data-path`: a path where it contains train/test.csv and image-folders
+ `--image-train-dir`: training images folder
+ `--image-test-dir`: test images folder
+ `--output-dir`: output dir for saving models

To train a model:
```
$ bash /path/to/project/run.sh
```
