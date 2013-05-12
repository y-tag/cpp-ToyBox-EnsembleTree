SRCTOP = ..

include $(SRCTOP)/Makefile.defs

CXXSRC = regression_tree.cc \
         random_forest_regressor.cc \
         gbrt.cc \
		 svmlight_reader.cc \

CXXFLAG += -fPIC -Wall

include $(SRCTOP)/Makefile.rules

lib:$(CXXOBJ)
	$(CXX) -shared -o libensemble.so $^ $(CXXFLAG)

clean:
	rm -f *~ *.o *.out *.so
