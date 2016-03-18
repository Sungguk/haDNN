//File: common.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "common.hh"
using namespace Halide;

namespace hadnn {

Image<float> random_image(const Shape& shape) {
	int d = shape.dim();
	switch (d) {
		case 1:
			{
				Image<float> ret(shape[0]);
				REP(i, shape[0]) ret(i) = rand() / RAND_MAX;
				return ret;
			}
		case 2:
			{
				Image<float> ret(shape[0], shape[1]);
				REP(i, shape[0])
					REP(j, shape[1]) ret(i, j) = rand() / RAND_MAX;
				return ret;
			}
		default:
			error_exit("Not Implemented Yet");
	}
}

}
