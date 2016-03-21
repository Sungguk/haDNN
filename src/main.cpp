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

/*
 *int main() {
 *  int B = 64 , Cin = 3, Cout = 64, H = 224, W = 224;
 *  ImageParam par(type_of<float>(), 4);
 *  par.set_bounds(0, 0, W);
 *  par.set_bounds(1, 0, H);
 *  par.set_bounds(2, 0, Cin);
 *  par.set_bounds(3, 0, B);
 *  par.set_stride(0, 1);
 *
 *
 *  Input input{par};
 *  auto paraW = random_image({3, 3, Cout, Cin}),
 *       parab = random_image({Cout});
 *  vector<Image<float>> params{paraW, parab};
 *  Conv2D l(&input, params, PaddingMode::SAME);
 *  l.default_sched();
 *
 *  speedtest_single_input(par, &l, {H, W, Cin, B}, {H, W, Cout, B});
 *}
 */

int main() {
	ImageParam par(type_of<float>(), 4);

	Input input{par};

	/*
	 *{ // conv2
	 *  int B = 64 , Cin = 64, Cout = 128, H = 112, W = 112;
	 *  auto paraW = random_image({3, 3, Cout, Cin}),
	 *       parab = random_image({Cout});
	 *  vector<Image<float>> params{paraW, parab};
	 *  Conv2DHWCN conv(&input, params, PaddingMode::SAME);
	 *  conv.default_sched();
	 *  speedtest_single_input(par, conv.get_output(), {B, Cin, W, H}, {B, Cout, W, H});
	 *}
	 */

	/*
	 *{ // pool1
	 *  int B = 64, C = 64, H = 224, W = 224;
	 *  Pooling maxpool(&input, {2, 2}, PaddingMode::VALID, PoolingMode::MAX);
	 *  maxpool.default_sched();
	 *  speedtest_single_input(par, &maxpool, {B, C, W, H}, {B, C, W/2, H/2});
	 *}
	 */

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
		net.get_output().print_loop_nest();

		speedtest_single_input(par, net.get_output(),
				{B, C[0], W, H}, {B, C[2], W/2, H/2});
	}
}
