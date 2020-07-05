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
```
#!/bin/bash
# to activate the virtual environment, change here
source [put your virtual environment path if you dont run it directly]

for fold in 0 1 2 3;
do
    python -m shopee.main \
          --backbone 'resnext101_32x8d_swsl' \
          --image-size 256 \
          --use-neck 0 \
          --criterion 'ce' \
          --fp16 False \
          --debug 0 \
          --data-path [a path where it contains train/test.csv and image-folders] \
          --image-train-dir [training images folder] \
          --image-test-dir [test images folder] \
          --batch-size 64 \
          --fold ${fold} \
          --lr 7e-6 \
          --epochs 30 \
          --device 0 \
          --output-dir [output dir for saving models]
done

```

To train a model:
```
$ bash /path/to/project/run.sh
```
