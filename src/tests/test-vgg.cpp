//File: test-vgg.cpp
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include <fstream>
#include "layers/everything.hh"
#include "common.hh"
#include "network.hh"
#include "testing.hh"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace Halide;
using namespace hadnn;
using namespace cv;

int main() {
	Mat im = imread("./cat.png");
	Mat imf;
	im.convertTo(imf, CV_32FC3);
	Mat imr;
	cv::resize(imf, imr, cv::Size(224, 224));
	REP(i, 224) REP(j, 224) imr.at<cv::Vec3f>(i, j) -= cv::Vec3f{110,110,110};
	auto imH = mat_to_image(imr, 8);
	auto params = read_params("vgg.tensortxt");

	ImageParam placeholder(type_of<float>(), 4);
	Input input{placeholder};

	int B = 8, H = 224, W = 224;

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

	auto out_img = random_image({B, 64, 112, 112});
	placeholder.set(imH);
	O.realize(out_img);

	ofstream fout("dump.tensortxt");
	write_tensor(out_img, fout);
}
