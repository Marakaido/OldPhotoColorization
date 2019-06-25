#!/bin/bash

TARGET_DIR=pytorch-CycleGAN-and-pix2pix/datasets/old-photos-dataset
SOURCE_DIR=$(pwd)

mkdir $TARGET_DIR
cd $TARGET_DIR

echo Downloading training dataset:
curl -# -o train.tar https://people.eecs.berkeley.edu/~nzhang/datasets/pipa_train.tar
tar -xvf train.tar
rm train.tar

echo Downloading test dataset:
curl -# -o test.tar https://people.eecs.berkeley.edu/~nzhang/datasets/pipa_test.tar
tar -xvf test.tar
rm test.tar

echo Downloading validation dataset:
curl -# -o val.tar https://people.eecs.berkeley.edu/~nzhang/datasets/pipa_val.tar
tar -xvf val.tar
rm val.tar

cd $SOURCE_DIR