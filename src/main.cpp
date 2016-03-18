//File: main.cpp
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include <vector>
#include "layers/softmax.hh"
#include "lib/utils.hh"
#include "lib/debugutils.hh"
#include "speed.hh"

using namespace std;
using namespace Halide;
using namespace hadnn;

int main() {
	ImageParam par(type_of<float>(), 2);

	Input input{par};
	SoftMax sm(&input);
	sm.default_sched();

	int n = 1000, m = 1000;
	speedtest_2D_input(par, &sm, {n, m}, {n, m});
	/*
	 *vector<Argument> args{par};
	 *output.compile_to_file("test", args);
	 */
}
