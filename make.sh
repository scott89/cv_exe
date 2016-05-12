g++ -I/usr/local/include -I/usr/include/python2.7 -I/home/lijun/anaconda2/lib/python2.7/site-packages/numpy/core/include -fpic filter.cpp -c -o filter.o

g++ -shared -Wl,-soname,"filter.so" -L/usr/local/lib filter.o -lboost_python -fpic -o filter.so

# g++ filter.cpp -c -o filter.o
# g++ test_filter.cpp filter.o -o test_filter
