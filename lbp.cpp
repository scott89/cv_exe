#include "PyCV.hpp"
using namespace std;
double* GetMap(const int samples) {
    assert(samples <=64);
    double* table = new double[(int)pow(2, samples)];
    for (unsigned long value = 0; value < pow(2, samples); value++) {
	unsigned long value_left = ((value & 1) << samples-1) | (value >> 1);
	int num_trans = std::bitset<64>(value_left ^ value).count();
	// cout << "value: " << (bitset<8>)value << " " << sum << endl;
	if (num_trans > 2) {
	    table[(int) value] = (double) (samples + 1);
	} else {
	    table[(int) value] = (double) (std::bitset<64>(value).count());
	}
	//cout << "value: " << (int)value << " " << (int) table[(int) value] << endl;

    }
    return table;
}

void ExtractImgLBP (const Img& in, Img& out, 
const double* map, const int channel, const int sample_num, const int r) {
    const int width = in.width();
    const int height = in.height();
    const int num_channel = in.channel();
    assert(num_channel > channel);
    out.Reshape(width - 2*r, height - 2*r, 1);
    const double* in_data = in.data() + channel * in.dim();
    double* out_data = out.mutable_data();
    for (int h = 0; h < out.height(); h++) {
	for (int w = 0; w < out.width(); w++) {
	    int h_ori = h + r;
	    int w_ori = w + r;
	    int lbp_code = 0;
	    for (int sample = 0; sample < sample_num; sample++) {
		double h_sample = r * std::sin(sample * 2 * PI / sample_num);
		double w_sample = r * std::cos(sample * 2 * PI / sample_num);
		int h_min = std::floor(h_sample);
		int h_max = std::floor(h_sample + 1);
		int w_min = std::floor(w_sample);
		int w_max = std::floor(w_sample + 1);
		double dw = w_sample - w_min;
		double dh = h_sample - h_min;
		double value = 
		in_data[(h_ori + h_min) * width + w_ori + w_min] * (r-dw) * (r-dh)/r/r +
		in_data[(h_ori + h_min) * width + w_ori + w_max] * (dw) * (r-dh) /r/r+
		in_data[(h_ori + h_max) * width + w_ori + w_min] * (r-dw) * (dh)/r/r +
		in_data[(h_ori + h_max) * width + w_ori + w_max] * (dw) * (dh)/r/r;
		lbp_code<<=1;
	        lbp_code += (int)(value > in_data[h_ori * width + w_ori]);  
	    }
	    out_data[h * out.width() + w] = map[lbp_code];
	}
    }
}


bp::object ExtractLBP(bp::object in_obj, bp::object channel_obj, bp::object lbp_obj) {
    Img in,  out;
    in.CopyFromPyArrayObject(reinterpret_cast<PyArrayObject*>(in_obj.ptr()));
    int sample = bp::extract<int>(lbp_obj[0]);
    int r = bp::extract<int>(lbp_obj[1]);
    int channel = bp::extract<int>(channel_obj);
    double* map = GetMap(sample);
    ExtractImgLBP(in, out, map, channel, sample, r);
    PyObject* out_obj =(PyObject*) out.ToPyArrayObject(); 
    bp::handle<> out_handle(out_obj);
    bp::numeric::array out_array(out_handle);
    return out_array.copy();
}

