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

		// setup default schedule
		virtual void default_sched() {}

		virtual ~Layer() {}

		Halide::Func& get_output() { return output_; }

	protected:
		std::vector<Layer*> tops_;
		std::vector<Halide::Image<float>> params_;

		Halide::Func output_{"output"};
};

enum class PaddingMode : char {
	VALID,
	SAME
};
}
