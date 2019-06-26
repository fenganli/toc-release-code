generate_mini_batches:
	g++ -std=c++14 generate_mini_batches.cc -O3 -o generate_mini_batches.o ../matrix/matrix.cc ../io/reader.cc ../compress/compress_data.h -lm -I $HOME/google-library/include -L $HOME/google-library/lib -lgflags -lglog -I. -L. -lsnappy -lboost_iostreams
clean:
	rm -rf *.o *.txt
