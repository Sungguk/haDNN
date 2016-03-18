//File: common.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once
#include <Halide.h>
#include "layers/shape.hh"
#include "lib/utils.hh"

namespace hadnn {

Halide::Image<float> random_image(const Shape& shape);


}
