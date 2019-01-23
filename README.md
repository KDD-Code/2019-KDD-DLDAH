Deep Discriminative Hashing Learning
===================

Part of the code is modified from [here](https://github.com/Minione/DTSH).

## Requirements ##
This code is written in MATLAB and requires [MatConvNet](http://www.vlfeat.org/matconvnet/).

## Preparation ##
- Download the CIFAR-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz). Uncompress the file and put the folder "cifar-10-batches-mat/" under the main folder.
- Download the Pretrained VGG-F model from [here](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat). Put the model under the main folder.

## Usage ##
Run the following command in MATLAB:
```
$ DLDAH(24, 10)
```


