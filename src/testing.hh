//File: testing.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once

#include <Halide.h>
#include <opencv2/core.hpp>

#include "layers/layer.hh"
#include "common.hh"
#include "lib/timer.hh"

namespace hadnn {

// return an image of HWCN
Halide::Image<float> mat_to_image(cv::Mat im, int batch_size) {
	Halide::Image<float> ret(batch_size, im.channels(), im.cols, im.rows);
	m_assert(im.channels() == 3);
	REP(i, im.rows) REP(j, im.cols)
		REP(n, batch_size) {
			cv::Vec3f c = im.at<cv::Vec3f>(i, j);
			REP(k, im.channels())
				ret(n, k, j, i) = c[k];
		}
	return ret;
}

void speedtest_single_input(
		Halide::ImageParam& input, Halide::Func& out_func,
		const Shape& in_shape, const Shape& out_shape) {
	auto in_img = random_image(in_shape);
	auto out_img = random_image(out_shape);
	input.set(in_img);
	out_func.compile_jit();
	{
		GuardedTimer tm("Realize Time");
		out_func.realize(out_img);
	}
};

void run_single_input(Halide::ImageParam& input, Layer* out_layer,
		const Halide::Image<float>& input_data, const Halide::Image<float>& output_data) {
	auto out_func = out_layer->get_output();
	input.set(input_data);
	out_func.realize(output_data);
}

}
