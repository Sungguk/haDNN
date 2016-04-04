//File: convfft.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once

#include "layer.hh"

namespace hadnn {

class Conv2DNCHWFFT : public Layer {
	public:
		// params: W, b
		// W: [ker_w, ker_h, out_ch, in_ch]
		// b: [out_ch]
		Conv2DNCHWFFT(Layer* top,
				const std::vector<Halide::Image<float>>& params,
				Shape in_shape,
				PaddingMode padding, Shape stride={1,1});

		void setup();

		int out_dim() const override { return 4; }

		ShapeExpr out_shape() const override;

		void default_sched() override;


		Halide::Func padded_kernel;
		Halide::Var Nidx{"Nidx"}, Cidx{"Cidx"}, Widx{"Widx"}, Hidx{"Hidx"};
	protected:
		PaddingMode padding_;
		Shape stride_, filter_, in_shape_;
		int out_ch_, in_ch_;
};

}
