#!/bin/sh

### Example usage:
### bash slice_infimnist 100
### Note 100 specifies the number of file shards.

mkdir -p ../data/mnist/infimnist
mkdir -p ../data/mnist/raw_data
cd ../infimnist
declare -i iter
iter=$1-1
for i in $(seq 0 $iter)
do
  declare -i start
  start=250*1000*$i+10000
  declare -i end
  end=250*1000*$i+250*1000+10000-1
  ./infimnist svm $start $end > ../data/mnist/raw_data/file-$i.libsvm
  echo $i is done
done
