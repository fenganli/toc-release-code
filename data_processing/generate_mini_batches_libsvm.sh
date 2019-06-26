#!/bin/sh

### Example usage:
### bash generate_generate_mini_batches_libsvm 100
### Note 100 specifies the number of file shards.

generate_libsvm_mini_batches_with_format () {
  ./generate_mini_batches.o --input_directory=${3} \
    --output_directory=${2} \
    --output_format=${1} \
    --mini_batch_size=250 \
    --num_mini_batches_per_shard=1000 \
    --start_shard_id=0 \
    --csv_reader=false \
    --end_shard_id=$4
}

mkdir -p ../data/mnist/mini_batches/
echo "generating mat format"
generate_libsvm_mini_batches_with_format mat ../data/mnist/mini_batches/ ../data/mnist/raw_data $1
echo "generating toc format"
generate_libsvm_mini_batches_with_format toc ../data/mnist/mini_batches/ ../data/mnist/raw_data $1
echo "generating csr format"
generate_libsvm_mini_batches_with_format csr ../data/mnist/mini_batches/ ../data/mnist/raw_data $1
echo "generating csrvi format"
generate_libsvm_mini_batches_with_format csrvi ../data/mnist/mini_batches/ ../data/mnist/raw_data $1
echo "generating dvi format"
generate_libsvm_mini_batches_with_format dvi ../data/mnist/mini_batches/ ../data/mnist/raw_data $1
echo "generating gzip format"
generate_libsvm_mini_batches_with_format gzip ../data/mnist/mini_batches/ ../data/mnist/raw_data $1
echo "generating snappy format"
generate_libsvm_mini_batches_with_format snappy ../data/mnist/mini_batches/ ../data/mnist/raw_data $1
