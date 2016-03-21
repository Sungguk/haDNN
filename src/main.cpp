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
		int C[]{3, 64, 128};
		int B = 64, H = 224, W = 224;
		auto paraW = random_image({3, 3, C[1], C[0]}),
				 parab = random_image({C[1]});
		vector<Image<float>> params{paraW, parab};
		vector<Image<float>> params2{
			random_image({3, 3, C[2], C[1]}), random_image({C[2]})};

		Network net(input);
		auto conv1 = new Conv2D(&input, params, PaddingMode::SAME);
		auto relu1 = new ReLU(conv1);
		auto pool1 = new Pooling(relu1, {2,2}, PaddingMode::VALID, PoolingMode::MAX);
		auto conv2 = new Conv2D(pool1, params2, PaddingMode::SAME);
		net.add(conv1).fence().add(relu1).fence().add(pool1).fence().add(conv2).fence();
		net.default_sched();

		auto& O = net.get_output();
		O.print_loop_nest();
		speedtest_single_input(par, net.get_output(),
				{B, C[0], W, H}, {B, C[2], W/2, H/2});
	}
}
