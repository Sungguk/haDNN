//File: pool.hh
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#pragma once

#include "layer.hh"

namespace hadnn {

enum class PoolingMode : char {
	MAX,
	AVG
};

// HWCN Pooling
class Pooling : public Layer {
	public:
		Pooling(Layer* top, Shape shape, PoolingMode mode, PaddingMode padding=PaddingMode::VALID) :
			Layer(top), shape_(shape), padding_(padding), mode_(mode) {
				m_assert(padding_ == PaddingMode::VALID);
				m_assert(mode_ == PoolingMode::MAX);
				setup();
			}

		void setup() {
			auto top = tops_.at(0);
			auto input = top->get_output();
			m_assert(input.dimensions() == 4);

			kernel = Halide::RDom{0, shape_[1], 0, shape_[0], "kernel"};
			// valid mode, no need to padding
			output_(Nidx, Cidx, Widx, Hidx)
				= Halide::maximum(input(Nidx, Cidx, Widx * shape_[1] + kernel.x,
							Hidx * shape_[0] + kernel.y));
		}

		void default_sched() override {
			Halide::Var par{"par"};
			output_.fuse(Hidx, Widx, par).parallel(par);
			output_.vectorize(Nidx, 8);
			//output_.print_loop_nest();
		}

		int out_dim() const override { return 4; }

		ShapeExpr out_shape() const override {
			auto top = tops_.at(0);
			auto in_shape = top->out_shape();
			in_shape[2] = in_shape[2] / shape_[1];
			in_shape[3] = in_shape[3] / shape_[0];
			return in_shape;
		}

		Halide::RDom kernel;
		Halide::Var Nidx{"Npool"}, Cidx{"Cpool"}, Hidx{"Hpool"}, Widx{"Wpool"};
	protected:
		Shape shape_;	// pooling stride in (h, w)
		PaddingMode padding_;
		PoolingMode mode_;
};
}	// namespace hadnn
