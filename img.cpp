#include "PyCV.hpp"

Img& Img::operator=(const Img &source){
    width_ = source.width();
    height_ = source.height();
    channel_ = source.channel();
    dim_ = source.dim();
    const double* source_data = source.data();
    if(data_ && !embed_) {
	free(data_);
    }
    data_ = new double[channel_ * dim_];
    embed_ = false;
    for (int c = 0; c < channel_; c++) {
	for (int h = 0; h < height_; h++) {
	    for (int w = 0; w < width_; w++) {
		data_[c * dim_ + h * width_ + w] = source_data[c * dim_ + h * width_ + w];
	    }
	}
    }
    return *this;
}
void Img::ReshapeLike(const Img & source) {
    if(width_!= source.width() || height_ != source.height() || channel_ != source.channel()) {
	width_ = source.width();
	height_ = source.height();
	channel_ = source.channel();
	dim_ = source.dim();
	if (data_ && !embed_) {
	    free(data_);
	}
	data_ = new double[dim_ * channel_];
	embed_ = false;
    }
}

void Img:: Reshape(const int w, const int h, const int c) {
    if(width_!= w || height_ != h || channel_ != c) {
	width_ = w;
	height_ = h;
	channel_ = c;
	dim_ = w * h;
	if (data_ && !embed_) {
	    free(data_);
	}
	data_ = new double[dim_ * channel_];
	embed_ = false;
    }
}

void Img::FromPyArrayObject(PyArrayObject * in) { 
    if(data_ && !embed_) {
	free(data_);
    }
    data_ = static_cast<double*>(PyArray_DATA(in));
    embed_ = true;
    int nd = PyArray_NDIM(in);
    npy_intp* dims = PyArray_DIMS(in);
    assert(nd >=1 && nd <= 3);
    height_ = dims[0]; 
    width_ = nd >=2? dims[1]: 1;
    channel_ = nd ==3? dims[2]:1;
    dim_ = width_ * height_;
}

void Img::CopyFromPyArrayObject(PyArrayObject * in ) {
    int nd = PyArray_NDIM(in);
    npy_intp* dims = PyArray_DIMS(in);
    assert(nd >=1 && nd <= 3);
    height_ = dims[0]; 
    width_ = nd >=2? dims[1]: 1;
    channel_ = nd ==3? dims[2]:1;
    dim_ = width_ * height_;
    double* in_data = static_cast<double*>(PyArray_DATA(in));
    if(data_ && !embed_) {
	free(data_);
    }
    data_ = new double[dim_ * channel_];
    embed_ = false;
    for(int c = 0; c < channel_; c++) {
	for(int h = 0; h < height_; h++) {
	    for(int w = 0; w < width_; w++) {
		data_[c * dim_ + h * width_ + w] = in_data[c * dim_ + h * width_ + w];
	    }
	}
    }
}

PyArrayObject* Img::ToPyArrayObject() { 
    assert(dim_ * channel_ > 0);
    int nd = 3;
    npy_intp dims [3];
    dims[0] = (npy_intp) height_; dims[1] = (npy_intp) width_; dims[2] = (npy_intp) channel_;  
    // The following two subroutines should be called before calling PyArray_SimpleNewFromData. Otherwise, an ugly segfualt will be caused.
    // see the follwing webpage for detailed expalaination: 
    // http://stackoverflow.com/questions/29851529/c-class-member-function-returns-pyobject-segmentation-fault 
    // import_array() and import_array1() does not work. I also try to include the follwing two lines in the constructor of Img class, but doesnt work either.
    Py_Initialize();
    _import_array();
    PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNewFromData(nd, dims, NPY_DOUBLE, data_);
    return out;
}



