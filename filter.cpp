#include "PyCV.hpp"

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

void FiltMedImg(const Img& in,  const int k_width, const int k_height, const int channel, Img& out) {
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
	    std::vector<double> temp_list;
	    for (int i = kh_start; i < kh_end; i++ ) {
		for (int j = kw_start; j < kw_end; j++) {
		    int w_temp = w_start + j;
		    int h_temp = h_start + i;
		    if (w_temp >=0 && w_temp < width && h_temp >= 0 && h_temp < height) {
			temp_list.push_back(in_data[h_temp * width + w_temp]);
		    }
		}
	    }
	    std::sort(temp_list.begin(), temp_list.end());
	    out_data[h * width + w] = temp_list[temp_list.size()/2];
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

bp::object FiltMed(bp::object in_obj, bp::object k_size, bp::object channel_obj) {
    Img in,  out;
    in.CopyFromPyArrayObject(reinterpret_cast<PyArrayObject*>(in_obj.ptr()));
    int k_height = bp::extract<int>(k_size[0]);
    int k_width = bp::extract<int>(k_size[1]);
    int channel = bp::extract<int>(channel_obj);
    FiltMedImg(in, k_width, k_height, channel, out);
    PyObject* out_obj =(PyObject*) out.ToPyArrayObject(); 
    bp::handle<> out_handle(out_obj);
    bp::numeric::array out_array(out_handle);
    return out_array.copy();

}


unsigned char* GetMap() {
    const int samples = 8;
    unsigned char* table = new unsigned char[(int)pow(2, samples)];
    for (unsigned char value = 0; value < pow(2, samples)-1; value++) {
        unsigned char value_left = ((value & 1) << samples-1) | (value >> 1);
        int num_trans = std::bitset<samples>(value_left ^ value).count();
       // cout << "value: " << (bitset<8>)value << " " << sum << endl;
        if (num_trans > 2) {
            table[(int) value] = (unsigned char) (samples + 1);
        } else {
            table[(int) value] = (unsigned char) (std::bitset<samples>(value).count());
        }
        //cout << "value: " << (int)value << " " << (int) table[(int) value] << endl;

    }
    return table;
}

