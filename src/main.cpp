//File: main.cpp
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include <vector>
#include "lib/utils.hh"
#include "lib/debugutils.hh"
#include "testing.hh"

#include "layers/everything.hh"
#include "network.hh"

using namespace std;
using namespace Halide;
using namespace hadnn;

vector<Image<float>> random_conv_param(int Cin, int Cout, int k) {
	return {random_image({k, k, Cout, Cin}), random_image({Cout})};
}

void speed_test_conv_nchw() {
	ImageParam par(type_of<float>(), 4);
	Input input{par};

	{	// vgg11
		int C[]{3, 64, 128, 256, 256, 512, 512, 512, 512};
		int B = 64, H = 224, W = 224;
		Sequential net(input);
		auto add_conv_pool = [&](int start_idx, int num_conv) {
			REP(k, num_conv) {
				net.add<Conv2DNCHW>(random_conv_param(
							C[start_idx+k], C[start_idx+k+1], 3), PaddingMode::SAME)
					 .add<ReLU>();
			}
		  net.add<PoolingNCHW>(Shape{2,2}, PoolingMode::MAX);
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
		P("NCHW:");
		speedtest_single_input(par, net.get_output(),
				{W, H, C[0], B}, {14, 14, 512,  B});
	}
}


void speed_test_conv_hwcn() {
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
		add_conv_pool(1, 1);	//1.06s,par
		 // 128x56x56
		add_conv_pool(2, 2);
		 // 256x28x28
		add_conv_pool(4, 2);
		 // 512x14x14
		add_conv_pool(6, 2);
		// 512x7x7

		//net.add<Conv2D>(random_conv_param(128, 256, 3), PaddingMode::SAME);	// 2.2s
		net.default_sched();
		auto& O = net.get_output();
		O.print_loop_nest();
		/*
		 *speedtest_single_input(par, net.get_output(),
		 *    {B, 128, 112, 112}, {B,256,112,112});
		 */
		P("HWCN:");
		speedtest_single_input(par, net.get_output(),
				{B, C[0], W, H}, {B,512,7,7});
	}
}

void speed_test_conv_nchw_fft() {
	ImageParam par(type_of<float>(), 4);
	Input input{par};

	int B = 64;
	int H = 128, W = 128;
	int Cin = 64, Cout = 64;
	Conv2DNCHWFFT l(&input, random_conv_param(Cin, Cout, 3),
			{H, W}, PaddingMode::SAME);

	l.default_sched();
	auto& O = l.get_output();
	//O.print_loop_nest();
	P("NCHWFFT:");
	speedtest_single_input(par, O, {W, H, Cin, B}, {W, H, Cout, B});
}

int main() {
//	speed_test_conv_hwcn();
//	speed_test_conv_nchw();
	speed_test_conv_nchw_fft();
}
