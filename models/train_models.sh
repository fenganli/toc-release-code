#!/bin/sh
echo 3 | sudo tee /proc/sys/vm/drop_caches

epoches=3
shards=1

echo Format:toc
./train_models.o --format=toc --model=ann --file_directory=../data/mnist/mini_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv
echo Format:mat
./train_models.o --format=mat --model=ann --file_directory=../data/mnist/mini_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv
echo Format:csr
./train_models.o --format=csr --model=ann --file_directory=../data/mnist/mini_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv
echo Format:csrvi
./train_models.o --format=csrvi --model=ann --file_directory=../data/mnist/mini_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv
echo Format:dvi
./train_models.o --format=dvi --model=ann --file_directory=../data/mnist/mini_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv
echo Format:snappy
./train_models.o --format=snappy --model=ann --file_directory=../data/mnist/mini_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv
echo Format:gzip
./train_models.o --format=gzip --model=ann --file_directory=../data/mnist/mini_batches --num_shards=${shards} --learning_rate=0.5 --num_epoches=${epoches} --test_file=../data/mnist/mnist_test.csv
