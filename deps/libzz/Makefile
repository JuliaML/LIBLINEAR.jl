CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC
# SHVER = 3
OS = $(shell uname)

lib: zz.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libzz.so"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libzz.so"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} zz.o -o libzz.so

zz.o: zz.cpp zz.h
	$(CXX) $(CFLAGS) -c -o zz.o zz.cpp

clean:
	rm -f *~ zz.o
