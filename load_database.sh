#!/bin/bash
# Usage:
# bash load_database.sh target [-0]
#   target - destination folder
#   -0 - optionally turn off data download

RAW_DATA_DIR=old-photos-raw-dataset
TARGET_DIR=$1

if [ $# = 2 ]; then
    mkdir $RAW_DATA_DIR
    cd $RAW_DATA_DIR

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

    cd ..
fi

rm -rf A
rm -rf B
mkdir A
mkdir A/train
mkdir A/test
mkdir A/val
mkdir B
mkdir B/train
mkdir B/test
mkdir B/val

python load_database.py $RAW_DATA_DIR

python pytorch-CycleGAN-and-pix2pix/datasets/combine_A_and_B.py --fold_A A --fold_B B --fold_AB $TARGET_DIR

rm -rf A
rm -rf B

cd $SOURCE_DIR