//File: network.hh
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#pragma once

#include <memory>
#include <vector>
#include "layers/layer.hh"
#include "layers/data.hh"

namespace hadnn {

class Network {
	public:
		Network(const std::vector<Input>& inputs):
			inputs_(inputs) {}
		Network(const Input& input):
			inputs_{input} {}

		// Network holds ownership of l
		Network& add(Layer* l) {
			layers_.emplace_back(l);
			fence();
			return *this;
		}

		Network& fence() {
			layers_.back()->get_output().compute_root();
			return *this;
		}

		Layer* last() {
			auto ret = layers_.back().get();
			m_assert(ret != nullptr);
			return ret;
		}

		void default_sched() {
			for (auto& l : layers_)
				l->default_sched();
		}

		Halide::Func& get_output() const {
			return layers_.back()->get_output();
		}

	protected:
		std::vector<Input> inputs_;
		std::vector<std::unique_ptr<Layer>> layers_;
};

class Sequential : public Network {
	public:
		Sequential(const Input& input): Network(input) { }

		template<typename T, typename... Args>
		Sequential& add(Args... args) {
			Layer* last = layers_.empty() ? &inputs_[0] : Network::last();
			Network::add(new T(last, args...));
			return *this;
		}
};

}
