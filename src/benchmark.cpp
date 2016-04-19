//File: benchmark.cpp
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

void benchmark_speed_test_conv_hwcn(int B, int H, int W, int k, int cin, int cout) {
	ImageParam par(type_of<float>(), 4);
	Input input{par};
	{
		Sequential net(input);
		net.add<Conv2D>(random_conv_param(cin, cout, k), PaddingMode::SAME);
		net.default_sched();
		// auto& O = net.get_output();
		// O.print_loop_nest();
		speedtest_single_input(par, net.get_output(),
				{B, cin, W, H}, {B, cout, W, H}, "ConvHWCN");
	}
}

void benchmark_speed_test_conv_nchw_fft(int B, int H, int W, int k, int cin, int cout) {
	ImageParam par(type_of<float>(), 4);
	Input input{par};

	// int B = 64;
	// int H = 48, W = 48;
	Conv2DNCHWFFT l(&input, random_conv_param(cin, cout, k),
			{H, W}, PaddingMode::SAME);

	l.default_sched();
	// auto& O = l.get_output();
	// O.print_loop_nest();

	speedtest_single_input(par, l.get_output(), {W, H, cin, B}, {W, H, cout, B}, "NCHWFFT");
}

int main(int argc, char const *argv[])
{
	if (argc != 8) {
		printf("Usage: <normal/fft> B H W k cin cout\n");
		return -1;
	}
	const char* type = argv[1];
	int B = atoi(argv[2]);
	int H = atoi(argv[3]);
	int W = atoi(argv[4]);
	int k = atoi(argv[5]);
	int cin = atoi(argv[6]);
	int cout = atoi(argv[7]);
	if (strcmp(type, "normal") == 0) {
		benchmark_speed_test_conv_hwcn(B, H, W, k, cin, cout);
	}
	if (strcmp(type, "fft") == 0) {
		benchmark_speed_test_conv_nchw_fft(B, H, W, k, cin, cout);
	}
	return 0;
}
