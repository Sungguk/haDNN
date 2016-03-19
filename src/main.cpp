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
	par.set_bounds(0, 0, H);
	par.set_bounds(1, 0, W);
	par.set_bounds(2, 0, Cin);
	par.set_bounds(3, 0, B);
	par.set_stride(0, 1);


	Input input{par};
	vector<Image<float>> params{random_image({3, 3, Cout, Cin}), random_image({Cout})};
	Conv2D l(&input, params, PaddingMode::SAME);
	l.default_sched();


	speedtest_single_input(par, &l, {H, W, Cin, B}, {H, W, Cout, B});
	/*
	 *vector<Argument> args{par};
	 *output.compile_to_file("test", args);
	 */
}
