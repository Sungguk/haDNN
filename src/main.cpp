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

	{ // conv2
		int B = 64 , Cin = 64, Cout = 128, H = 112, W = 112;
		auto paraW = random_image({3, 3, Cout, Cin}, "W"),
				 parab = random_image({Cout}, "b");
		vector<Image<float>> params{paraW, parab};
		auto conv = new Conv2D(&input, params, PaddingMode::SAME);
		Network net(input); net.add(conv).fence();
		net.default_sched();
		auto& O = net.get_output();
		speedtest_single_input(par, O, {B, Cin, W, H}, {B, Cout, W, H});
	}

	{	// conv1 + pool1 + conv2
		int C[]{3, 64, 128, 256, 256, 512, 512, 512, 512};
		int B = 64, H = 224, W = 224;
		Network net(input);
		auto conv1 = new Conv2D(&input, random_conv_param(C[0], C[1], 3), PaddingMode::SAME);
		auto relu1 = new ReLU(conv1);
		auto pool1 = new Pooling(relu1, {2,2}, PoolingMode::MAX);
		auto conv2 = new Conv2D(pool1, random_conv_param(C[1], C[2], 3), PaddingMode::SAME);
		auto relu2 = new ReLU(conv2);
		auto pool2 = new Pooling(relu2, {2,2}, PoolingMode::MAX);
		// 128x56x56
		auto conv3 = new Conv2D(pool2, random_conv_param(C[2], C[3], 3), PaddingMode::SAME);
		auto relu3 = new ReLU(conv3);
		auto conv4 = new Conv2D(relu3, random_conv_param(C[3], C[4], 3), PaddingMode::SAME);
		auto relu4 = new ReLU(conv4);
		auto pool3 = new Pooling(relu4, {2,2}, PoolingMode::MAX);
		// 256x28x28
		auto conv5 = new Conv2D(pool3, random_conv_param(C[4], C[5], 3), PaddingMode::SAME);
		auto relu5 = new ReLU(conv5);
		auto conv6 = new Conv2D(relu5, random_conv_param(C[5], C[6], 3), PaddingMode::SAME);
		auto relu6 = new ReLU(conv6);
		auto pool4 = new Pooling(relu6, {2,2}, PoolingMode::MAX);
		// 512x14x14
		auto conv7 = new Conv2D(pool4, random_conv_param(C[5], C[6], 3), PaddingMode::SAME);
		auto relu7 = new ReLU(conv7);
		auto conv8 = new Conv2D(relu7, random_conv_param(C[6], C[7], 3), PaddingMode::SAME);
		auto relu8 = new ReLU(conv8);
		auto pool5 = new Pooling(relu8, {2,2}, PoolingMode::MAX);
		net.add(conv1).add(relu1).add(pool1)
			 .add(conv2).add(relu2).add(pool2)
			 .add(conv3).add(relu3).add(conv4).add(relu4).add(pool3)
			 .add(conv5).add(relu5).add(conv6).add(relu6).add(pool4)
			 .add(conv7).add(relu7).add(conv8).add(relu8).add(pool5);
		net.default_sched();

		auto& O = net.get_output();
		O.print_loop_nest();
		speedtest_single_input(par, net.get_output(),
				{B, C[0], W, H}, {B, 512, 7, 7});
	}
}
