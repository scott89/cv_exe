// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <boost/python.hpp>
//#include <Python.h>
//#include <boost/python/module.hpp>
//#include <boost/python/def.hpp>
#include <numpy/arrayobject.h>
using namespace boost::python;
//namespace bp = boost::python;


class Img {
    public:
    Img(): width_(0), height_(0), channel_(0), dim_(0), data_(NULL) {}
    Img(int w, int h, int c): width_(w), height_(h), channel_(c), dim_(w*h) {
        data_ = new double[channel_ * dim_];
    }
    Img(const Img& source): width_(source.width()), height_(source.height()),
    channel_(source.channel()), dim_(source.dim()) {
        data_ = new double[channel_ * dim_]
        source_data = source.data();
        for (int c = 0; c < channel_; c++) {
            for (int h = 0; h < height_; h++) {
                for (int w = 0; w < width_; w++) {
                    data_[c * dim_ + h * width_ + w] = source_data[c * dim_ + h * width_ + w];
                }
            }
        }
    }

    int width() {
        return width_;
    }
    int height() {
        return height_;
    }
    int channel() {
        return channel_;
    }
    int dim() {
        return dim_;
    }



object filter(object x) {
    PyArrayObject* pt = reinterpret_cast<PyArrayObject*>(x.ptr());
    double* dt = static_cast<double*>(PyArray_DATA(pt));
    int nd = PyArray_NDIM(pt);
    npy_intp* dims = PyArray_DIMS(pt);
    int n_el = 1;
    for (int i = 0; i < nd; i++) {
        n_el *= dims[i];
    }
    printf("%d\n", n_el);
    double* tdt = new double[n_el];
    //double* tdt = new double(9);
    std::memcpy(tdt, dt, n_el * sizeof(double));
    tdt[0] = 0;
    
    printf("%d\n", n_el * sizeof(double));
    //tdt[0] = 0;
    PyArrayObject* o = (PyArrayObject *) PyArray_SimpleNewFromData(nd, dims, NPY_DOUBLE, tdt);
    handle<> hdl((PyObject*)o);
    numeric::array arr(hdl);
    return arr.copy();
}

BOOST_PYTHON_MODULE(test)
{
    numeric::array::set_module_and_type("numpy", "ndarray"); 
    def("filter", filter);
    import_array1();
}
