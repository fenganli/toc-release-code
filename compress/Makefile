compress_test:
	g++ -std=c++0x compress_test.cc -O3 -o compress_test.o ../matrix/matrix.cc ../io/reader.cc -lm -I $HOME/google-library/include -L $HOME/google-library/lib -lgflags -lglog -I. -L. -lsnappy -lboost_iostreams

clean:
	rm -rf compress_test.o *.o *.txt
