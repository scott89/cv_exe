// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <boost/python.hpp>
//#include <Python.h>
//#include <boost/python/module.hpp>
//#include <boost/python/def.hpp>
#include <numpy/arrayobject.h>
//using namespace boost::python;
namespace bp = boost::python;
//using namespace boost::python;
#include <memory> // header files for shared_ptr
#include <stdio.h> 
#include <stdlib.h>
#include <cassert> // header files for assert
#include <vector> // header files for vector
#include <algorithm> // header files for sort
#include <math.h>
#include <bitset>
#define MAX_DOUBLE 999999999
#define PI 3.14159265
class Img {
 public:
    Img(): width_(0), height_(0), channel_(0), dim_(0), data_(NULL), embed_(false) { }
    Img(int w, int h, int c): width_(w), height_(h), channel_(c), dim_(w*h), embed_(false) {
        assert(w*h*c);
        data_ = new double[channel_ * dim_];
    }
    /*
    Notes:
    1. The copy constructor is only used for object construction. After construction, the assignment operation use the automatically generated operator rather than the copy constructor. The automatically generated assignment operator directly copys the data_ pointer, which leads to double freement when deconstruction.
    2. the data_ pointer is randomly initialized instead of initialied to NULL, if it is not explicitly listed in the initialization list.
    3. PyArray_SimpleNewFramData creates and return a "copy" rather than a "reference" of the data pointed by the pointer passed into the function. 
    4. PyArray_DATA returns a refenrece of the data embeded in PyArrayObject. That is, if we can modify the numpy array in python by directly modifying the data pointed by the returned pointer in C++.
    5. Notice double free of the data pointer in deconstruction function.
    */
    Img(const Img& source): width_(source.width()), height_(source.height()),
    channel_(source.channel()), dim_(source.dim()), data_(), embed_(false) {
        const double* source_data = source.data();
        data_ = new double[channel_ * dim_];
        for (int c = 0; c < channel_; c++) {
            for (int h = 0; h < height_; h++) {
                for (int w = 0; w < width_; w++) {
                    data_[c * dim_ + h * width_ + w] = source_data[c * dim_ + h * width_ + w];
                }
            }
        }
    }
    ~Img() {
	if (data_ && !embed_) {
        //printf("%p\n", data_);
	    free(data_);
        }
        data_ = NULL;
    }
    void FromPyArrayObject(PyArrayObject *);
    void CopyFromPyArrayObject(PyArrayObject *);
    PyArrayObject* ToPyArrayObject();

    Img& operator=(const Img& );
    void ReshapeLike(const Img&);
    void Reshape(const int, const int, const int);

    int width() const {
        return width_;
    }
    int height() const {
        return height_;
    }
    int channel() const {
        return channel_;
    }
    int dim() const {
        return dim_;
    }
    const double* data() const{
	return data_;
    }
    double* mutable_data() {
        return data_;
    } 

 protected:
    int width_;
    int height_;
    int channel_;
    int dim_;
    double* data_;
    bool embed_; // indicating wheter the data_ pointer is pointed to a numpy data blob. If it is, the data_ pointer cannot be freed, otherwise, the numpy array will be freed either.
    //double* data_;
};

void FiltImg(const Img&, const Img&,  const int, Img&);
void FiltMaxImg(const Img&, const int,  const int, const int, Img&);
void FiltMedImg(const Img&, const int, const int, const int, Img&);
double* GetMap(const int = 8);
void ExtractImgLBP(const Img&, Img&, const double*, const int = 0, const int = 8, const int = 1);


bp::object Filt(bp::object, bp::object, bp::object);
bp::object FiltMax(bp::object, bp::object, bp::object);
bp::object FiltMed(bp::object, bp::object, bp::object);
bp::object ExtractLBP(bp::object, bp::object, bp::object);

