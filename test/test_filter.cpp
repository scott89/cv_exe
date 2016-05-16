#include "filter.hpp"
#include <iostream>
using namespace std;
Img aa(Img x) {
    cout << "x: " << x.mutable_data() << endl;
    cout << "in aa, width = " << x.width() << endl;;
  //  Img y = x;
  //  cout << "y: " << y.data_ << endl;
    return x;
}

void InputImg(Img& in) {
    cout << "input Imag W x H x C: " << in.width() << "x" << in.height() << "x"  << in.channel() << endl;
    double* data = in.mutable_data();
    for (int c = 0; c < in.channel(); c++) {
        for (int h = 0; h < in.height(); h++) {
            for (int w = 0; w < in.width(); w++) {
                cin >> data[c * in.dim() + h * in.width() + w];
            }
        }
    }
}
void PrintImg(const Img& in) {
const double* data = in.data();
    cout << "Img W x H x C: " << in.width() << "x" << in.height() << "x"  << in.channel() << endl;
    for(int h = 0; h < in.height(); h++) {
	for (int w = 0; w < in.width(); w++) {
	    cout << data[h * in.width() + w] << " ";
	}
	cout << endl;
    }
}

int main() {
    Img im2(4,4,1);
    Img im(2,2,1);
    Img f(2,2,1);
    im.ReshapeLike(im2);
    InputImg(im);
    im2 = im;
    InputImg(f);

    //Filt(im2, f, 0, im);
    FiltMax(im2, 2, 2, 0, im);
    PrintImg(im2);
    PrintImg(im); 
	    
    return 1;
}

