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

void write_tensor(const Halide::Image<float>& v, std::ostream& os);

std::unordered_map<std::string, Halide::Image<float>> read_params(std::string fname);

}
