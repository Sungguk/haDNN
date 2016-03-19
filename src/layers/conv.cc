//File: conv.cc
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include "conv.hh"

using namespace Halide;

namespace hadnn {

Conv2DNCHW::Conv2DNCHW(Layer* top,
				const std::vector<Halide::Image<float>>& params,
				PaddingMode padding, Shape stride):
	Layer(top),
	padding_(padding), stride_(stride)
{
	params_ = params;
	m_assert(params.size() == 2UL);
	m_assert(params_[0].dimensions() == 4);
	m_assert(params_[1].dimensions() == 1);
	m_assert(stride_[0] == 1 && stride_[1] == 1);
	setup();
}

void Conv2DNCHW::setup() {
	auto top = tops_.at(0);
	auto in_shape = top->out_shape();

	auto &W = params_[0], &b = params_[1];
	filter_ = {W.extent(0), W.extent(1)};
	out_ch_ = {W.extent(2)};
	in_ch_ = W.extent(3);
	m_assert(b.extent(0) == out_ch_);

	auto input = top->get_output();
	m_assert(input.dimensions() == 4);

	if (padding_ == PaddingMode::SAME) {
		padded = BoundaryConditions::constant_exterior(input, 0,
				{{0, in_shape[0]}, {0, in_shape[1]},
				 {0, in_shape[2]}, {0, in_shape[3]}});

		kernel = RDom{0, filter_[0], 0, filter_[1], 0, in_ch_, "kernel"};
		output_(Widx, Hidx, Cidx, Nidx) = b(Cidx);
		output_(Widx, Hidx, Cidx, Nidx) +=
			W(kernel.x, kernel.y, Cidx, kernel.z) *
			padded(Widx + kernel.y - filter_[1]/2, Hidx + kernel.x - filter_[0]/2,
						 kernel.z, Nidx);
	}
}

void Conv2DNCHW::default_sched() {
	padded.compute_root();
	Var par{"par"};
	output_.vectorize(Widx, 8);
	output_.fuse(Nidx, Cidx, par).parallel(par);
	output_.compute_root();

	auto&& U = output_.update();
	U.reorder(Widx, Hidx, kernel.z);

	Var Wo{"Wo"}, Wi{"Wi"}, Co{"Co"}, Ci{"Ci"}, Ho{"Ho"}, Hi{"Hi"};
	U.vectorize(Widx, 8);
	if (out_ch_ >= 32) {
		U.split(Cidx, Co, Ci, 16);
		U.fuse(Nidx, Co, par).parallel(par);
	} else {
		U.parallel(Nidx);
	}
	output_.print_loop_nest();
}

ShapeExpr Conv2DNCHW::out_shape() const {
	auto top = tops_.at(0);
	auto in_shape = top->out_shape();
	in_shape[2] = out_ch_;
	if (padding_ == PaddingMode::VALID) {
		in_shape[0] = in_shape[0] - filter_[0] + 1;
		in_shape[1] = in_shape[1] - filter_[1] + 1;
	}
	return in_shape;
}

}  // namespace hadnn
