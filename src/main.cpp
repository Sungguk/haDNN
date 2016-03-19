//File: main.cpp
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include <vector>
#include "lib/utils.hh"
#include "lib/debugutils.hh"
#include "testing.hh"

#include "layers/softmax.hh"
#include "layers/conv.hh"

using namespace std;
using namespace Halide;
using namespace hadnn;

int main() {
	int B = 64 , Cin = 3, Cout = 64, H = 224, W = 224;
	ImageParam par(type_of<float>(), 4);

	Input input{par};
	vector<Image<float>> params{random_image({Cin, Cout, 3, 3}), random_image({Cout})};
	Conv2D l(&input, params, PaddingMode::SAME);
	l.default_sched();


	speedtest_single_input(par, &l, {H, W, Cin, B}, {H, W, Cout, B});
	/*
	 *vector<Argument> args{par};
	 *output.compile_to_file("test", args);
	 */
}
