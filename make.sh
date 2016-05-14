g++ -I/usr/local/include -I/usr/include/python2.7 -I/home/lijun/anaconda2/lib/python2.7/site-packages/numpy/core/include -fpic filter.cpp -c -o filter.o
g++ -I/usr/local/include -I/usr/include/python2.7 -I/home/lijun/anaconda2/lib/python2.7/site-packages/numpy/core/include -fpic img.cpp -c -o img.o
g++ -I/usr/local/include -I/usr/include/python2.7 -I/home/lijun/anaconda2/lib/python2.7/site-packages/numpy/core/include -fpic lbp.cpp -c -o lbp.o
g++ -I/usr/local/include -I/usr/include/python2.7 -I/home/lijun/anaconda2/lib/python2.7/site-packages/numpy/core/include -fpic PyCV.cpp -c -o PyCV.o
g++ -shared -Wl,-soname,"PyCV.so" -L/usr/local/lib filter.o img.o lbp.o PyCV.o -lboost_python -fpic -o PyCV.so

#g++ -shared -Wl,-soname,"PyCV.so" -L/usr/local/lib filter.o img.o PyCV.o -lboost_python -fpic -o PyCV.so




# g++ filter.cpp -c -o filter.o
# g++ test_filter.cpp filter.o -o test_filter
