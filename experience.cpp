#include "filter.hpp"
#include <iostream>
using namespace std;
/*
1. when direct assign one Img instance to another, the data pointer is also directly copyed.
2. passing an Img instance as parameter to a function does not seems to an assignment, since the data potiner is different. However, when return an Img instance from function, it is again a assignment operation.
*/
Img aa(Img x) {
    cout << "x: " << x.data_ << endl;
    cout << "in aa, width = " << x.width() << endl;;
    Img y = x;
    cout << "y: " << y.data_ << endl;
    return y;
}
int main() {
    Img im2(4,4,1);
    cout << "im2: " << im2.data_ << endl;
    Img im = im2;
    cout << "im: " << im.data_ << endl;
    im = aa(im2);
    cout << "im: " << im.data_ << endl;


    cout << im.width() << endl;
    const double* data = im.data();
    for(int h = 0; h < im.height(); h++) {
	for (int w = 0; w < im.width(); w++) {
	    cout << data[h * im.width() + w] << " ";
	}
	cout << endl;
    }
	    
    return 1;
}

