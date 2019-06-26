train_models:
	g++ -std=c++14 train_models.cc -O3 -o train_models.o ../matrix/matrix.cc ../io/reader.cc ../compress/compress_data.h -lm -I $HOME/google-library/include -L $HOME/google-library/lib -lgflags -lglog -I. -L. -lsnappy -lboost_iostreams -march=native -ffast-math -funroll-loops -fopenmp -D_GLIBCXX_PARALLEL

clean:
	rm -rf *.o
