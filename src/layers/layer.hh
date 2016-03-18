//File: layer.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once
#include <vector>
#include <Halide.h>
#include "lib/debugutils.hh"

class TensorShape {
	public:
		TensorShape(Halide::Expr a): shape_{a, -1, -1, -1} {}
		TensorShape(Halide::Expr a, Halide::Expr b): shape_{a, b, -1, -1} {}
		TensorShape(Halide::Expr a, Halide::Expr b, Halide::Expr c): shape_{a, b, c, -1} {}
		TensorShape(Halide::Expr a, Halide::Expr b, Halide::Expr c, Halide::Expr d): shape_{a, b, c, d} {}

		Halide::Expr at(int k) const {
			m_assert(k < 4 && k >= 0);
			return shape_[k];
		}

		Halide::Expr operator [](int k) const { return shape_[k]; }
	private:
		Halide::Expr shape_[4];
};

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
		virtual TensorShape out_shape() const = 0;

		virtual void setup() = 0;

		virtual ~Layer() {}

		const Halide::Func& get_forward() const { return forward_; }


	protected:
		std::vector<Layer*> tops_;
		std::vector<Halide::Image<float>> params_;

		Halide::Func forward_;
};

class Input : public Layer {
	public:
		Input(Halide::ImageParam param): Layer(nullptr), inputs_(param) {}

		void setup() override {
			ndim_ = inputs_.dimensions();
			switch (ndim_) {
				case 1:
					forward_(x) = inputs_(x);
					break;
				case 2:
					forward_(x, y) = inputs_(x, y);
					break;
				case 3:
					forward_(x, y, z) = inputs_(x, y, z);
					break;
				case 4:
					forward_(x, y, z, w) = inputs_(x, y, z, w);
					break;
				default:
					error_exit("Unknown dimension");
			}
		}

		TensorShape out_shape() const override {
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
