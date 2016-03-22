//File: test-vgg.cpp
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include "layers/everything.hh"
#include "common.hh"
#include "network.hh"
#include "testing.hh"

using namespace Halide;
using namespace hadnn;

int main() {
	auto params = read_params("tests/vgg.tensortxt");

	ImageParam placeholder(type_of<float>(), 4);
	Input input{placeholder};

	int B = 10, H = 224, W = 224;

	Network net(input);
	auto conv11 = new Conv2D(&input,
			{params["conv1_1/W"], params["conv1_1/b"]},
			PaddingMode::SAME);
	auto relu11 = new ReLU(conv11);
	auto conv12 = new Conv2D(relu11,
			{params["conv1_2/W"], params["conv1_2/b"]}, PaddingMode::SAME);
	auto relu12 = new ReLU(conv12);
	auto pool1 = new Pooling(relu12, {2, 2}, PaddingMode::VALID, PoolingMode::MAX);
	// 64, 112, 112

	net.add(conv11).add(relu11).add(conv12).add(relu12).add(pool1);
	net.default_sched();

	auto& O = net.get_output();
	O.print_loop_nest();
	speedtest_single_input(placeholder, O, {B, 3, W, H}, {B, 64, 112, 112});
}
