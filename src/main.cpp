//File: main.cpp
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include <vector>
#include "lib/utils.hh"
#include "lib/debugutils.hh"
#include "testing.hh"

#include "layers/softmax.hh"
#include "layers/conv.hh"
#include "layers/pool.hh"
#include "layers/nonlin.hh"
#include "layers/data.hh"
#include "network.hh"

using namespace std;
using namespace Halide;
using namespace hadnn;

vector<Image<float>> random_conv_param(int Cin, int Cout, int k) {
	return {random_image({k, k, Cout, Cin}), random_image({Cout})};
}

int main() {
	ImageParam par(type_of<float>(), 4);

	Input input{par};

	{	// vgg11
		int C[]{3, 64, 128, 256, 256, 512, 512, 512, 512};
		int B = 64, H = 224, W = 224;
		Sequential net(input);
		auto add_conv_pool = [&](int start_idx, int num_conv) {
			REP(k, num_conv) {
				net.add<Conv2D>(random_conv_param(C[start_idx+k], C[start_idx+k+1], 3), PaddingMode::SAME)
					 .add<ReLU>();
			}
		  net.add<Pooling>(Shape{2,2}, PoolingMode::MAX);
		};
		add_conv_pool(0, 1);
		add_conv_pool(1, 1);
		// 128x56x56
		add_conv_pool(2, 2);
		// 256x28x28
		add_conv_pool(4, 2);
		// 512x14x14
		add_conv_pool(6, 2);
		// 512x7x7
		net.default_sched();

		auto& O = net.get_output();
		O.print_loop_nest();
		speedtest_single_input(par, net.get_output(),
				{B, C[0], W, H}, {B, 512, 7, 7});
	}
}
