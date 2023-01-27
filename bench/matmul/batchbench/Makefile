################################################################################
# Copyright 2020 Intel Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

ifndef MKLROOT
    $(warning MKLROOT is not set. Run make as `MKLROOT=/path/to/mkl make`)
endif

CXXFLAGS += -Wall -Wextra
CXXFLAGS += -std=c++11

CPPFLAGS += -DNDEBUG -D_FORTIFY_SOURCE=2
CPPFLAGS_MKL += $(CPPFLAGS)
CXXFLAGS += -O3 -fpermissive -g

AFFINITY := KMP_AFFINITY=compact,1,0,granularity=fine

CXXFLAGS += -I${MKLROOT}/include

LDLIBS_MKL += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
LDFLAGS += -L${MKLROOT}/lib/intel64 -L${MKLROOT}/../compiler/lib/intel64

.PHONY: run clean

mkl_exe   := bench

run: $(mkl_exe)
	LD_LIBRARY_PATH="${MKLROOT}/lib/intel64:${MKLROOT}/../compiler/lib/intel64:$$LD_LIBRARY_PATH" \
			$(AFFINITY) ./$(mkl_exe)

$(mkl_exe): bench.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS_MKL)

bench.o: bench.cpp timer.h
	$(CXX) -o $@ $(CPPFLAGS_MKL) $(CXXFLAGS) -c $<

clean:
	$(RM) *.o $(mkl_exe)
