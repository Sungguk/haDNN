//File: common.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "common.hh"
using namespace Halide;

namespace hadnn {

Image<float> random_image(const Shape& shape, string name) {
	int d = shape.dim();
	switch (d) {
		case 1:
			{
				Image<float> ret(shape[0], name);
				REP(i, shape[0]) ret(i) = rand() * 1.0 / RAND_MAX;
				return ret;
			}
		case 2:
			{
				Image<float> ret(shape[0], shape[1], name);
				REP(i, shape[0])
					REP(j, shape[1]) ret(i, j) = rand() * 1.0 / RAND_MAX;
				return ret;
			}
		case 3:
			{
				Image<float> ret(shape[0], shape[1], shape[2], name);
				REP(i, shape[0])
					REP(j, shape[1])
						REP(k, shape[2]) ret(i, j, k) = rand() * 1.0 / RAND_MAX;
				return ret;
			}
		case 4:
			{
				Image<float> ret(shape[0], shape[1], shape[2], shape[3], name);
				REP(i, shape[0])
					REP(j, shape[1])
						REP(k, shape[2])
							REP(l, shape[3]) ret(i, j, k, l) = rand() * 1.0 / RAND_MAX;
				return ret;
			}
		default:
			error_exit("Unsupported Dimension");
	}
}

void write_tensor(const Image<float>& v, std::ostream& os) {
	os << const_cast<Image<float>&>(v).name();
	for (int d = v.dimensions()-1; d >= 0; d--)
		os << " " << v.extent(d);
	os << endl;
	switch (v.dimensions()) {
		case 1:
			REP(i, v.extent(0))
				os.write((char*)&v(i), sizeof(float));
			break;
		case 2:
			REP(j, v.extent(1))
				REP(i, v.extent(0))
					os.write((char*)&v(i, j), sizeof(float));
			break;
		case 3:
			REP(k, v.extent(2))
				REP(j, v.extent(1))
					REP(i, v.extent(0))
					  os.write((char*)&v(i, j, k), sizeof(float));
			break;
		case 4:
			REP(l, v.extent(3))
				REP(k, v.extent(2))
					REP(j, v.extent(1))
						REP(i, v.extent(0))
					    os.write((char*)&v(i, j, k, l), sizeof(float));
			break;
	}
}

}
