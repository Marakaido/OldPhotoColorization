#!/bin/bash
# Usage:
# bash load_database.sh raw_data_folder target_folder

RAW_DATA_DIR=$1

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

python process_raw_database.py $RAW_DATA_DIR

rm -rf $2
mkdir $2
python pytorch-CycleGAN-and-pix2pix/datasets/combine_A_and_B.py --fold_A A --fold_B B --fold_AB $2

rm -rf A
rm -rf B