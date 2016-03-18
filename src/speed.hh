//File: speed.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once

#include <Halide.h>
#include "layers/layer.hh"
#include "common.hh"
#include "lib/timer.hh"

namespace hadnn {

void speedtest_2D_input(
		Halide::ImageParam& input, Layer* out_layer,
		const Shape& in_shape, const Shape& out_shape) {
	auto out_func = out_layer->get_output();

	auto in_img = random_image(in_shape);
	auto out_img = random_image(out_shape);
	input.set(in_img);
	out_func.realize(out_img);	// compile
	{
		GuardedTimer tm("2D realize");
		out_func.realize(out_img);
	}
};

}
