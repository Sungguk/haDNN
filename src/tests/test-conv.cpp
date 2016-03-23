//File: test-conv.cpp
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include <vector>
#include <fstream>
#include "lib/utils.hh"
#include "lib/debugutils.hh"
#include "common.hh"
#include "layers/conv.hh"
#include "layers/data.hh"

using namespace std;
using namespace Halide;
using namespace hadnn;
/*
 *
 *int test_conv_nchw() {
 *  int B = 64 , Cin = 3, Cout = 64, H = 224, W = 224;
 *  ImageParam par(type_of<float>(), 4);
 *
 *  Input input{par};
 *  auto paraW = random_image({3, 3, Cout, Cin}, "W"),
 *       parab = random_image({Cout}, "b");
 *  vector<Image<float>> params{paraW, parab};
 *  Conv2DNCHW l(&input, params, PaddingMode::SAME);
 *  l.default_sched();
 *
 *  auto out_func = l.get_output();
 *  auto in_img = random_image({W, H, Cin, B}, "input");
 *  auto out_img = random_image({W, H, Cout, B}, "output");
 *  par.set(in_img);
 *  out_func.realize(out_img);
 *
 *  ofstream fout("conv_nchw_out.txt");
 *  write_tensor(paraW, fout);
 *  write_tensor(parab, fout);
 *  write_tensor(in_img, fout);
 *  write_tensor(out_img, fout);
 *}
 */

int test_conv_hwcn() {
	int B = 8 , Cin = 4, Cout = 16, H = 224, W = 224;
	ImageParam par(type_of<float>(), 4);

	Input input{par};
	auto paraW1 = random_image({3, 3, Cout, Cin}, "W1"),
			 parab1 = random_image({Cout}, "b1"),
			 paraW2 = random_image({3, 3, Cout, Cout}, "W2"),
			 parab2 = random_image({Cout}, "b2");
	Conv2DHWCN l(&input, {paraW1, parab1}, PaddingMode::SAME);
	Conv2DHWCN l2(&l, {paraW2, parab2}, PaddingMode::SAME);
	l.get_output().compute_root();
	l.default_sched();
	l2.default_sched();

	auto out_func = l2.get_output();
	out_func.print_loop_nest();
	auto in_img = random_image({B, Cin, W, H}, "input");
	auto out_img = random_image({B, Cout, W, H}, "output");
	par.set(in_img);
	out_func.realize(out_img);

	ofstream fout("conv_hwcn_out.txt");
	write_tensor(paraW1, fout);
	write_tensor(parab1, fout);
	write_tensor(paraW2, fout);
	write_tensor(parab2, fout);
	write_tensor(in_img, fout);
	write_tensor(out_img, fout);
}

int main() {
	test_conv_hwcn();
}
