#include "PyCV.hpp"
BOOST_PYTHON_MODULE(PyCV)
{
    bp::numeric::array::set_module_and_type("numpy", "ndarray"); 
    def("Filt", Filt);
    def("FiltMax", FiltMax);
    def("FiltMed", FiltMed);
    import_array1();
}
