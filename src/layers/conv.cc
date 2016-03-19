//File: conv.cc
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include "conv.hh"

using namespace Halide;

namespace hadnn {

Conv2DHWCN::Conv2DHWCN(Layer* top,
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

void Conv2DHWCN::setup() {
	auto top = tops_.at(0);
	auto in_shape = top->out_shape();

	auto &W = params_[0], &b = params_[1];
	filter_ = {W.extent(2), W.extent(3)};
	out_ch_ = {W.extent(1)};
	int in_ch = W.extent(0);
	m_assert(b.extent(0) == out_ch_);

	auto input = top->get_output();
	m_assert(input.dimensions() == 4);

	if (padding_ == PaddingMode::SAME) {
		Func padded = BoundaryConditions::constant_exterior(input, 0,
				{{0, in_shape[0]}, {0, in_shape[1]},
				 {0, in_shape[2]}, {0, in_shape[3]}});
		padded.compute_root();

		kernel = RDom{0, filter_[0], 0, filter_[1], 0, in_ch, "kernel"};
		output_(Hidx, Widx, Cidx, Nidx) = b(Cidx);
		output_(Hidx, Widx, Cidx, Nidx) +=
			W(kernel.z, Cidx, kernel.x, kernel.y) *
			padded(Hidx + kernel.x - filter_[0]/2, Widx + kernel.y - filter_[1]/2,
					   kernel.z, Nidx);
	}
}

void Conv2DHWCN::default_sched() {
	Var NC;
	output_.fuse(Nidx, Cidx, NC).parallel(NC);
	output_.compute_root();

	auto&& U = output_.update();
	U.reorder(Widx, Hidx, kernel.z);

	Var Wo{"Wo"}, Wi{"Wi"}, Co{"Co"}, Ci{"Ci"}, Ho{"Ho"}, Hi{"Hi"};
	U.unroll(kernel.x).unroll(kernel.y);
	U.vectorize(Hidx, 8);
	U.split(Cidx, Co, Ci, 16);
	U.fuse(Nidx, Co, NC).parallel(NC);
	//output_.print_loop_nest();
}

ShapeExpr Conv2DHWCN::out_shape() const {
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
