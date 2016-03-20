//File: data.hh
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#pragma once
#include "layer.hh"

namespace hadnn {

class Input : public Layer {
	public:
		Input(Halide::ImageParam param): Layer(nullptr), inputs_(param)
		{ setup(); }

		void setup() {
			ndim_ = inputs_.dimensions();
			switch (ndim_) {
				case 1:
					output_(x) = inputs_(x);
					break;
				case 2:
					output_(x, y) = inputs_(x, y);
					break;
				case 3:
					output_(x, y, z) = inputs_(x, y, z);
					break;
				case 4:
					output_(x, y, z, w) = inputs_(x, y, z, w);
					break;
				default:
					error_exit("Unknown dimension");
			}
		}

		ShapeExpr out_shape() const override {
			switch (ndim_) {
				case 1:
					return {inputs_.extent(0)};
				case 2:
					return {inputs_.extent(0), inputs_.extent(1)};
				case 3:
					return {inputs_.extent(0), inputs_.extent(1), inputs_.extent(2)};
				case 4:
					return {inputs_.extent(0), inputs_.extent(1), inputs_.extent(2), inputs_.extent(3)};
				default:
					error_exit("This shouldn't happen");
			}
		}

		int out_dim() const override { return ndim_; }

	protected:
		Halide::ImageParam inputs_;
		int ndim_;
		Halide::Var x,y,z,w;
};

}
