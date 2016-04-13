//File: convfft.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "convfft.hh"
#include "fft/fft.h"
#include "lib/utils.hh"
#include "lib/debugutils.hh"

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
	fft_shape_ = Shape{pow2roundup(in_shape_[0] + W.extent(1)/2),
										 pow2roundup(in_shape_[1] + W.extent(0)/2)};
	print_debug("FFT shape: %d, %d\n", fft_shape_[0], fft_shape_[1]);
	if (max(fft_shape_[0], fft_shape_[1]) >= 256) {
		print_debug("FFT for shape >= 256 might be buggy.\n");
	}
	m_assert(b.extent(0) == out_ch_);

	auto top = tops_.at(0);
	ShapeExpr in_shape_expr = top->out_shape();
	auto input = top->get_output();
	m_assert(input.dimensions() == 4);

	// see doc, maybe don't need this
	Func Wfunc;
	Wfunc(Widx, Hidx, Coutidx, Cidx) = W(W.extent(0)-1-Widx, W.extent(1)-1-Hidx, Coutidx, Cidx);
	Func Wpadded = BoundaryConditions::constant_exterior(Wfunc, 0,
			{{0, filter_[1]}, {0, filter_[0]}, {0, out_ch_}, {0, in_ch_}});

	Func padded = BoundaryConditions::constant_exterior(input, 0,
			{{0, in_shape_[1]}, {0, in_shape_[0]}, {0, in_ch_}, {0, in_shape_expr[3]}});
	auto target = get_jit_target_from_environment();

	Fft2dDesc desc;
	desc.name = "imgfft";
	img_fft = fft2d_r2c(padded, fft_shape_[1], fft_shape_[0], target, desc);
	desc.name = "Wfft";
	W_fft = fft2d_r2c(Wpadded, fft_shape_[1], fft_shape_[0], target);

	rv = RDom{0, in_ch_, "rv"};
	cgemm(Widx, Hidx, Coutidx, Nidx) = ComplexExpr{0, 0};
	cgemm(Widx, Hidx, Coutidx, Nidx) += img_fft(Widx, Hidx, rv.x, Nidx) * W_fft(Widx, Hidx, Coutidx, rv.x);

	desc.gain = 1.0f / (fft_shape_[0] * fft_shape_[1]);
	desc.name = "ifft";
	ifft = fft2d_c2r(cgemm, fft_shape_[1], fft_shape_[0], target, desc);
	output_(Widx, Hidx, Cidx, Nidx) = ifft(Widx + W.extent(0)/2, Hidx + W.extent(1)/2, Cidx, Nidx) + b(Cidx);
}

void Conv2DNCHWFFT::default_sched() {
	// TODO parallel
	W_fft.compute_root();
	img_fft.compute_at(cgemm, Nidx);

	auto&& U = cgemm.update();
	U.reorder(Widx, Hidx, rv.x, Coutidx, Nidx).vectorize(Widx, 8);

	cgemm.compute_at(ifft, Nidx);
	ifft.compute_at(output_, Nidx);
}

ShapeExpr Conv2DNCHWFFT::out_shape() const {
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
