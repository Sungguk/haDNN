//File: convfft.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "convfft.hh"
#include "fft/fft.h"

using namespace Halide;

namespace hadnn {

#define CHECK_CONV_PARAMS \
	m_assert(params.size() == 2UL); \
	m_assert(params_[0].dimensions() == 4); \
	m_assert(params_[1].dimensions() == 1); \
	m_assert(stride_[0] == 1 && stride_[1] == 1)

Conv2DNCHWFFT::Conv2DNCHWFFT(Layer* top,
				const std::vector<Halide::Image<float>>& params,
				Shape in_shape,
				PaddingMode padding, Shape stride):
	Layer(top),
	padding_(padding), stride_(stride), in_shape_(in_shape) {
		params_ = params;
		CHECK_CONV_PARAMS;
		m_assert(padding == PaddingMode::SAME);
		setup();
	}

void Conv2DNCHWFFT::setup() {
	auto &W = params_[0], &b = params_[1];
	filter_ = {W.extent(1), W.extent(0)};
	out_ch_ = W.extent(2);
	in_ch_ = W.extent(3);
	m_assert(b.extent(0) == out_ch_);

	auto top = tops_.at(0);
	auto input = top->get_output();
	m_assert(input.dimensions() == 4);

	// see doc, maybe don't need this
	Func Wfunc; Wfunc(Nidx, Cidx, Widx, Hidx) = W(Nidx, Cidx, Widx, Hidx);
	Func padded_kernel = BoundaryConditions::constant_exterior(
			Wfunc, 0,
			{{0, filter_[1]}, {0, filter_[0]},
			{0, out_ch_}, {0, in_ch_}});	// XXX wrong offset
	auto target = get_jit_target_from_environment();
	auto kernel_fft = fft2d_r2c(padded_kernel, in_shape_[1], in_shape_[0], target);

	auto img_fft = fft2d_r2c(input, in_shape_[1], in_shape_[0], target);
	img_fft.compute_root();
	ComplexFunc mult("fft_mult");
	mult(Widx, Hidx, Cidx, Nidx) = kernel_fft(Widx, Hidx, Cidx, Nidx) * img_fft(Widx, Hidx, Cidx, Nidx);

	mult.compute_root();

	Fft2dDesc inv_desc;
	inv_desc.gain = 1.0f / (in_shape_[0] * in_shape_[1]);

	output_ = fft2d_c2r(img_fft, in_shape_[1], in_shape_[0], target, inv_desc);
}

void Conv2DNCHWFFT::default_sched() {

}

ShapeExpr Conv2DNCHWFFT::out_shape() const {
	// XXX copied
	auto top = tops_.at(0);
	auto in_shape = top->out_shape();
	in_shape[2] = out_ch_;
	if (padding_ == PaddingMode::VALID) {
		in_shape[0] = in_shape[0] - filter_[1] + 1;
		in_shape[1] = in_shape[1] - filter_[0] + 1;
	}
	return in_shape;
}

}
