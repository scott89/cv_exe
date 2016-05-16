CC := g++
INCLUDES_DIR := /usr/local/include \
        /usr/include/python2.7 \
        /home/lijun/anaconda2/lib/python2.7/site-packages/numpy/core/include
LIB_DIR := /usr/local/lib
LIB := boost_python
OBJS := filter.o img.o lbp.o PyCV.o
INCLUDE_FLAGS := $(foreach in_dir, $(INCLUDES_DIR), -I$(in_dir))
LIB_FLAGS := $(foreach lib_dir, $(LIB_DIR), -L$(lib_dir)) 
LIB_FLAGS += $(foreach lib, $(LIB), -l$(lib))
CCFLAGS := $(INCLUDE_FLAGS) -fpic
#LINKFLAGS := -shared -Wl,-soname,"PyCV.so" $(LIB_FLAGS) -fpic
LINKFLAGS := -shared $(LIB_FLAGS) -fpic

.PHONY: all clean cleanobj cleanso


all: PyCV.so 

PyCV.so: $(OBJS)
	@echo $(CC) $@
	@$(CC)  $(OBJS) $(LINKFLAGS) -o $@
%o: %cpp
	@echo $(CC) $@
	@$(CC) $(CCFLAGS) $< -c -o $@ 

clean: cleanobj cleanso

cleanobj:
	rm *.o
cleanso:
	rm *.so
        
print:
	@echo $(LINKFLAGS)
