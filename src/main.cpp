//File: main.cpp
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include <vector>
#include "layers/softmax.hh"
#include "lib/utils.hh"
#include "lib/debugutils.hh"

using namespace std;
using namespace Halide;

int main() {
	ImageParam par(type_of<float>(), 2);

	Input input{par};
	input.setup();
	SoftMax sm(&input);
	sm.setup();

	auto output = sm.get_forward();

	Image<float> in(10, 10), out(10, 10);
	REP(i, 10) REP(j, 10) in(i, j) = i + j;
	par.set(in);
	output.realize(out);
	REP(i, 10) REP(j, 10)
		print_debug("%d, %d=%lf\n", i, j, out(i, j));
	/*
	 *vector<Argument> args{par};
	 *output.compile_to_file("test", args);
	 */
}
