mkdir $1
cd $1

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