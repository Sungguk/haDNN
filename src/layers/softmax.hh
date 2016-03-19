//File: softmax.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once

#include "layer.hh"

namespace hadnn {

// batched softmax, output a prob distribution for each sample in the batch
// input: BxN
// output: BxN
class SoftMax : public Layer {
	public:
		SoftMax(Layer* top): Layer(top) { setup(); }

		int out_dim() const override { return 2; }

		void setup() {
			auto top = tops_.at(0);
			auto in_shape = top->out_shape();
			auto input = top->get_output();

			// TODO numeric stability
			Halide::RDom sum_reduction(0, in_shape[1]);
			exp_input(batch_idx, cls_idx) = exp(input(batch_idx, cls_idx));
			sum_exp(batch_idx) = sum(exp_input(batch_idx, sum_reduction));
			output_(batch_idx, cls_idx) = exp_input(batch_idx, cls_idx) / sum_exp(batch_idx);
		}

		void default_sched() override {
			// TODO optimize more
			exp_input.vectorize(cls_idx, 8)
							 .compute_root();

			sum_exp.vectorize(batch_idx, 8)
						 .compute_root();

			output_.vectorize(cls_idx, 8)
						 .compute_root();
		}

		ShapeExpr out_shape() const override { return tops_.at(0)->out_shape(); }


		Halide::Func exp_input{"exp_input"}, sum_exp{"sum_exp"};
		Halide::Var batch_idx, cls_idx;
};

}
