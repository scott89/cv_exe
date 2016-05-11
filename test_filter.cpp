#include "filter.hpp"
#include <iostream>
using namespace std;
int main() {
    Img im2(4,4,1);
    Img im = im2;

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

