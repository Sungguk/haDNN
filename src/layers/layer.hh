//File: layer.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once
#include <vector>
#include <Halide.h>
#include "shape.hh"
#include "lib/debugutils.hh"

namespace hadnn {

// base class of all layers
class Layer {
	public:
		Layer(std::vector<Layer*> tops):
			tops_(tops) { }

		Layer(Layer* top) {
			if (top != nullptr)
				tops_ = {top};
		}

		virtual int out_dim() const = 0;	// output dimension, at most 4 (limited by Halide)
		virtual ShapeExpr out_shape() const = 0;

		virtual ~Layer() {}

		const Halide::Func& get_output() const { return output_; }


	protected:
		std::vector<Layer*> tops_;
		std::vector<Halide::Image<float>> params_;

		Halide::Func output_;
};

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
