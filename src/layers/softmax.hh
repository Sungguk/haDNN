//File: softmax.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once

#include "layer.hh"

// batched softmax, output a prob distribution for each sample in the batch
// input: BxN
// output: BxN
class SoftMax : public Layer {
	public:
		SoftMax(Layer* top): Layer(top) {}

		int out_dim() const override { return 2; }

		void setup() override {
			auto top = tops_.at(0);
			auto in_shape = top->out_shape();

			auto input = top->get_forward();


			// TODO numeric stability
			Halide::RDom sum_reduction(0, in_shape[1]);
			Halide::Func exp_input;
			exp_input(batch_idx, cls_idx) = exp(input(batch_idx, cls_idx));
			Halide::Func sum_exp;
			sum_exp(batch_idx) = sum(exp_input(batch_idx, sum_reduction));
			sum_exp.trace_stores();
			forward_(batch_idx, cls_idx) = exp_input(batch_idx, cls_idx) / sum_exp(batch_idx);
		}

		TensorShape out_shape() const override {
			return tops_.at(0)->out_shape();
		}

		Halide::Var batch_idx, cls_idx;
};
