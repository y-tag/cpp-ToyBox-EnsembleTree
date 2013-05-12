SRCTOP = ..

include $(SRCTOP)/Makefile.defs

INCLUDES += -I../lib \

LIBS += -L../lib \
        -Wl,-rpath,../lib \
        -lensemble \

include $(SRCTOP)/Makefile.rules

all:eval-regression

eval-regression:eval_regression.cc
	$(CXX) -o $@ $^ $(CFLAG) $(INCLUDES) $(LIBS)

clean:
	rm -f *~ *.o eval-regression

