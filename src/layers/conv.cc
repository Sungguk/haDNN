//File: conv.cc
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include "conv.hh"
#include "lib/utils.hh"

using namespace Halide;

namespace hadnn {

#define CHECK_CONV_PARAMS \
	m_assert(padding_ == PaddingMode::SAME); \
	m_assert(params.size() == 2UL); \
	m_assert(params_[0].dimensions() == 4); \
	m_assert(params_[1].dimensions() == 1); \
	m_assert(stride_[0] == 1 && stride_[1] == 1)

/*
 *Conv2DNCHW::Conv2DNCHW(Layer* top,
 *        const std::vector<Halide::Image<float>>& params,
 *        PaddingMode padding, Shape stride):
 *  Layer(top),
 *  padding_(padding), stride_(stride)
 *{
 *  params_ = params;
 *  CHECK_CONV_PARAMS;
 *  setup();
 *}
 *
 *void Conv2DNCHW::setup() {
 *  auto top = tops_.at(0);
 *  auto in_shape = top->out_shape();
 *
 *  auto &W = params_[0], &b = params_[1];
 *  filter_ = {W.extent(0), W.extent(1)};
 *  out_ch_ = W.extent(2);
 *  in_ch_ = W.extent(3);
 *  m_assert(b.extent(0) == out_ch_);
 *
 *  auto input = top->get_output();
 *  m_assert(input.dimensions() == 4);
 *
 *  if (padding_ == PaddingMode::SAME) {
 *    padded = BoundaryConditions::constant_exterior(input, 0,
 *        {{0, in_shape[0]}, {0, in_shape[1]},
 *         {0, in_shape[2]}, {0, in_shape[3]}});
 *
 *    kernel = RDom{0, filter_[0], 0, filter_[1], 0, in_ch_, "kernel"};
 *    output_(Widx, Hidx, Cidx, Nidx) = b(Cidx);
 *    output_(Widx, Hidx, Cidx, Nidx) +=
 *      W(kernel.x, kernel.y, Cidx, kernel.z) *
 *      padded(Widx + kernel.x - filter_[1]/2, Hidx + kernel.y - filter_[0]/2,
 *             kernel.z, Nidx);
 *  }
 *}
 *
 *void Conv2DNCHW::default_sched() {
 *  padded.compute_root();
 *  Var par{"par"};
 *  output_.vectorize(Widx, 8);
 *  output_.fuse(Nidx, Cidx, par).parallel(par);
 *  output_.compute_root();
 *
 *  auto&& U = output_.update();
 *  U.reorder(Widx, Hidx, kernel.z);
 *
 *  Var Wo{"Wo"}, Wi{"Wi"}, Co{"Co"}, Ci{"Ci"}, Ho{"Ho"}, Hi{"Hi"};
 *  U.vectorize(Widx, 8);
 *  if (out_ch_ >= 32) {
 *    U.split(Cidx, Co, Ci, 16);
 *    U.fuse(Nidx, Co, par).parallel(par);
 *  } else {
 *    U.parallel(Nidx);
 *  }
 *  output_.print_loop_nest();
 *}
 *
 *ShapeExpr Conv2DNCHW::out_shape() const {
 *  auto top = tops_.at(0);
 *  auto in_shape = top->out_shape();
 *  in_shape[2] = out_ch_;
 *  if (padding_ == PaddingMode::VALID) {
 *    in_shape[0] = in_shape[0] - filter_[0] + 1;
 *    in_shape[1] = in_shape[1] - filter_[1] + 1;
 *  }
 *  return in_shape;
 *}
 *
 */
Conv2DHWCN::Conv2DHWCN(Layer* top,
				const std::vector<Halide::Image<float>>& params,
				PaddingMode padding, Shape stride):
	Layer(top),
	padding_(padding), stride_(stride)
{
	params_ = params;
	CHECK_CONV_PARAMS;
	setup();
}

void Conv2DHWCN::setup() {
	auto top = tops_.at(0);
	ShapeExpr in_shape = top->out_shape();
	auto &W = params_[0], &b = params_[1];
	filter_ = {W.extent(1), W.extent(0)};
	out_ch_ = W.extent(2);
	in_ch_ = W.extent(3);
	m_assert(b.extent(0) == out_ch_);

	auto& input = top->get_output();
	m_assert(input.dimensions() == 4);

	if (padding_ == PaddingMode::SAME) {
		padded = BoundaryConditions::constant_exterior(input, 0,
				{{0, in_shape[0]}, {0, in_ch_},
				 {0, in_shape[2]}, {0, in_shape[3]}});

		kernel = RDom{0, filter_[1], 0, filter_[0], 0, in_ch_, "kernel"};
		output_(Nidx, Cidx, Widx, Hidx) = b(Cidx);
		output_(Nidx, Cidx, Widx, Hidx) +=
				W(kernel.x, kernel.y, Cidx, kernel.z) *
				padded(Nidx,kernel.z,Widx+kernel.x-1,Hidx+kernel.y-1);
	}
}

void Conv2DHWCN::default_sched() {
	Var par{"par"};
	output_.fuse(Hidx, Widx, par).parallel(par);

	auto&& U = output_.update();
	U.reorder(Nidx, kernel.z);
	U.reorder(Cidx, kernel.z);

	U.vectorize(Nidx, 8);
	if (true) {
		// parallel
		U.fuse(Hidx, Widx, par).parallel(par);
		padded.compute_at(output_, par);
	} else {
		padded.compute_at(output_, Widx);
	}
	//output_.print_loop_nest();
}

ShapeExpr Conv2DHWCN::out_shape() const {
	auto top = tops_.at(0);
	ShapeExpr in_shape = top->out_shape();
	in_shape[1] = out_ch_;
	if (padding_ == PaddingMode::VALID) {
		in_shape[3] = in_shape[3] - filter_[0] + 1;
		in_shape[2] = in_shape[2] - filter_[1] + 1;
	}
	return in_shape;
}


}  // namespace hadnn
