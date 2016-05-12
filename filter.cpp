#include "filter.hpp"

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
    PyArrayObject* out = (PyArrayObject *) PyArray_SimpleNewFromData(nd, dims, NPY_DOUBLE, data_);
    return out;
}

void FiltImg(const Img& in, const Img& filter, const int channel, Img& out) {
    const int width = in.width();
    const int height = in.height();
    const int num_channel = in.channel();
    assert(num_channel > channel);
    out.Reshape(width, height, 1);
    const double* in_data = in.data() + channel * out.dim();
    const double* filter_data = filter.data();
    double* out_data = out.mutable_data();
    const int k_width = filter.width();
    const int k_height = filter.height();
    for(int h = 0; h < height; h++) {
	for (int w = 0; w < width; w++) {
	    int w_start = w - ((k_width) / 2);
	    int h_start = h - ((k_height) / 2);
	    int kw_start = 0;
	    int kh_start = 0;
	    int kw_end = k_width;
	    int kh_end = k_height;
	    double sum = 0;
	    for (int i = kh_start; i < kh_end; i++ ) {
		for (int j = kw_start; j < kw_end; j++) {
		    int w_temp = w_start + j;
		    int h_temp = h_start + i;
		    if (w_temp >=0 && w_temp < width && h_temp >= 0 && h_temp < height) {
			sum += in_data[h_temp * width + w_temp] * filter_data[i * k_width + j];
                       //printf("%f x %f, ", in_data[h_temp * width + w_temp],  filter_data[i * k_width + j]);
                        //printf("%d %d, \n", h_temp, w_temp);
		    }
		}
	    }
	    out_data[h * width + w] = sum;
	}
    }

}

void FiltMaxImg(const Img& in,  const int k_width, const int k_height, const int channel, Img& out) {
    const int width = in.width();
    const int height = in.height();
    const int num_channel = in.channel();
    assert(num_channel > channel);
    out.Reshape(width, height, 1);
    const double* in_data = in.data();
    double* out_data = out.mutable_data();

    for(int h = 0; h < height; h++) {
	for (int w = 0; w < width; w++) {
	    int w_start = w - ((k_width) / 2);
	    int h_start = h - ((k_height) / 2);
	    int kw_start = 0;
	    int kh_start = 0;
	    int kw_end = k_width;
	    int kh_end = k_height;
	    double max_value = -MAX_DOUBLE;
	    for (int i = kh_start; i < kh_end; i++ ) {
		for (int j = kw_start; j < kw_end; j++) {
		    int w_temp = w_start + j;
		    int h_temp = h_start + i;
		    if (w_temp >=0 && w_temp < width && h_temp >= 0 && h_temp < height) {
			max_value = std::max(max_value, in_data[h_temp * width + w_temp]);
		    }
		}
	    }
	    out_data[h * width + w] = max_value;
	}
    }

}

bp::object Filt(bp::object in_obj, bp::object kernel_obj, bp::object channel_obj) {
    Img in, kernel, out;
    in.FromPyArrayObject(reinterpret_cast<PyArrayObject*>(in_obj.ptr()));
    kernel.FromPyArrayObject(reinterpret_cast<PyArrayObject*>(kernel_obj.ptr()));
    int channel = bp::extract<int>(channel_obj);
    FiltImg(in, kernel, channel, out);
    PyObject* out_obj =(PyObject*) out.ToPyArrayObject(); 
    bp::handle<> out_handle(out_obj);
    bp::numeric::array out_array(out_handle);
    return out_array.copy();
}

bp::object FiltMax(bp::object in_obj, bp::object k_size, bp::object channel_obj) {
    Img in,  out;
    in.CopyFromPyArrayObject(reinterpret_cast<PyArrayObject*>(in_obj.ptr()));
    int k_height = bp::extract<int>(k_size[0]);
    int k_width = bp::extract<int>(k_size[1]);
    int channel = bp::extract<int>(channel_obj);
    FiltMaxImg(in, k_width, k_height, channel, out);
    PyObject* out_obj =(PyObject*) out.ToPyArrayObject(); 
    bp::handle<> out_handle(out_obj);
    bp::numeric::array out_array(out_handle);
    return out_array.copy();

}

BOOST_PYTHON_MODULE(filter)
{
    bp::numeric::array::set_module_and_type("numpy", "ndarray"); 
    def("Filt", Filt);
    def("FiltMax", FiltMax);
    import_array1();
}
