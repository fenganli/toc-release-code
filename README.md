This is guide for how to play around with TOC.

Step1, start a machine:
In Google cloud, create a ubuntu 16.04 instance with 500 GB boot disk, 15 GB memory, and 4 CPUs.

Step2, install some dependencies:
$ sudo apt-get update
$ sudo apt-get install vim
$ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
$ sudo apt-get -y install libsnappy-dev
$ sudo apt-get -y install libboost-all-dev
$ sudo apt-get -y install python3-pip
$ sudo pip3 install absl-py
$ sudo pip3 install matplotlib==2.1.1

Step3, get the data
$ cd ~/
$ git clone https://github.com/fenganli/toc-release-code.git

Step4, get infimnist
$ cd ~/
$ wget https://leon.bottou.org/_media/projects/infimnist.tar.gz
$ tar -xvzf infimnist.tar.gz

Step5, generate the data:
You can first generate a small dataset with 5 file shards:
$ cd ~/toc-release-code/data_processing
$ make generate_mini_batches
$ bash slice_infimnist.sh 5
$ bash generate_mini_batches_libsvm 5

If you want to see the full power of toc, you can generate a large dataset with 100 file
shards, it may take a while though.

Step6, train a neural network using mini-batch gradient descent powered by TOC:
$ cd ~/toc-release-code/models
$ make train_models
$ bash train_models.sh
