//File: conv.hh
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#pragma once
#include "layer.hh"

namespace hadnn {

enum class PaddingMode : char {
	VALID,
	SAME
};

// NCHW Conv2D
class Conv2D : public Layer {
	public:
		// params: W, b
		// W: [in_ch, out_ch, ker_w, ker_h]
		// b: [out_ch]
		Conv2D(Layer* top,
				const std::vector<Halide::Image<float>>& params,
				PaddingMode padding, Shape stride={1,1});

		void setup();

		int out_dim() const override { return 4; }

		ShapeExpr out_shape() const override;

		void default_sched() override;

		Halide::Var Nidx{"Nidx"}, Cidx{"Cidx"}, Hidx{"Hidx"}, Widx{"Widx"};
	protected:
		PaddingMode padding_;
		Shape stride_, filter_;
		int out_ch_;
};

}		// namespace hadnn
