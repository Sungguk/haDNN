//File: common.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once
#include <Halide.h>
#include <iostream>
#include <unordered_map>
#include "layers/shape.hh"
#include "lib/utils.hh"

namespace hadnn {

Halide::Image<float> random_image(const Shape& shape, std::string name="");

// return a 2D tensor in shape [W, H]
Halide::Image<float> read_img2d(string fname, int H, int W);

// return a 3D tensor in shape [W, H, 3]
Halide::Image<float> read_img3d(string fname, int H, int W);

// return a 4D tensor in shape [W, H, 3, N]
Halide::Image<float> read_img4d_n3hw(string fname, int H, int W, int N, int ch=3);

std::unordered_map<std::string, Halide::Image<float>> read_params(std::string fname);

template <typename T>
void write_tensor(const Halide::Image<T>& v, std::ostream& os) {
	os << const_cast<Halide::Image<T>&>(v).name();
	for (int d = v.dimensions()-1; d >= 0; d--)
		os << " " << v.extent(d);
	os << std::endl;
	switch (v.dimensions()) {
		case 1:
			REP(i, v.extent(0))
				os.write((char*)&v(i), sizeof(T));
			break;
		case 2:
			REP(j, v.extent(1))
				REP(i, v.extent(0))
					os.write((char*)&v(i, j), sizeof(T));
			break;
		case 3:
			REP(k, v.extent(2))
				REP(j, v.extent(1))
					REP(i, v.extent(0))
					  os.write((char*)&v(i, j, k), sizeof(T));
			break;
		case 4:
			REP(l, v.extent(3))
				REP(k, v.extent(2))
					REP(j, v.extent(1))
						REP(i, v.extent(0))
					    os.write((char*)&v(i, j, k, l), sizeof(T));
			break;
	}
}

}
