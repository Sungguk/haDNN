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
	auto conv21 = new Conv2D(pool1,
			{params["conv2_1/W"], params["conv2_1/b"]}, PaddingMode::SAME);
	auto relu21 = new ReLU(conv21);
	auto conv22 = new Conv2D(relu21,
			{params["conv2_2/W"], params["conv2_2/b"]}, PaddingMode::SAME);
	auto relu22 = new ReLU(conv22);
	auto pool2 = new Pooling(relu22, {2, 2}, PaddingMode::VALID, PoolingMode::MAX);
	// 128, 56, 56
	auto conv31 = new Conv2D(pool2,
			{params["conv3_1/W"], params["conv3_1/b"]}, PaddingMode::SAME);
	auto relu31 = new ReLU(conv31);
	auto conv32 = new Conv2D(relu31,
			{params["conv3_2/W"], params["conv3_2/b"]}, PaddingMode::SAME);
	auto relu32 = new ReLU(conv32);
	auto conv33 = new Conv2D(relu32,
			{params["conv3_3/W"], params["conv3_3/b"]}, PaddingMode::SAME);
	auto relu33 = new ReLU(conv33);
	auto pool3 = new Pooling(relu33, {2, 2}, PaddingMode::VALID, PoolingMode::MAX);
	// 256, 28, 28
	auto conv41 = new Conv2D(pool3,
			{params["conv4_1/W"], params["conv4_1/b"]}, PaddingMode::SAME);
	auto relu41 = new ReLU(conv41);
	auto conv42 = new Conv2D(relu41,
			{params["conv4_2/W"], params["conv4_2/b"]}, PaddingMode::SAME);
	auto relu42 = new ReLU(conv42);
	auto conv43 = new Conv2D(relu42,
			{params["conv4_3/W"], params["conv4_3/b"]}, PaddingMode::SAME);
	auto relu43 = new ReLU(conv43);
	auto pool4 = new Pooling(relu43, {2, 2}, PaddingMode::VALID, PoolingMode::MAX);
	// 512, 14, 14
	auto conv51 = new Conv2D(pool4,
			{params["conv5_1/W"], params["conv5_1/b"]}, PaddingMode::SAME);
	auto relu51 = new ReLU(conv51);
	auto conv52 = new Conv2D(relu51,
			{params["conv5_2/W"], params["conv5_2/b"]}, PaddingMode::SAME);
	auto relu52 = new ReLU(conv52);
	auto conv53 = new Conv2D(relu52,
			{params["conv5_3/W"], params["conv5_3/b"]}, PaddingMode::SAME);
	auto relu53 = new ReLU(conv53);
	auto pool5 = new Pooling(relu53, {2, 2}, PaddingMode::VALID, PoolingMode::MAX);
	// 512, 7, 7

	net.add(conv11).add(relu11).add(conv12).add(relu12).add(pool1)
		 .add(conv21).add(relu21).add(conv22).add(relu22).add(pool2)
		 .add(conv31).add(relu31).add(conv32).add(relu32).add(conv33).add(relu33).add(pool3)
		 .add(conv41).add(relu41).add(conv42).add(relu42).add(conv43).add(relu43).add(pool4)
		 .add(conv51).add(relu51).add(conv52).add(relu52).add(conv53).add(relu53).add(pool5);
	net.default_sched();

	auto& O = net.get_output();
	O.print_loop_nest();

	auto out_img = random_image({B, 512, 7, 7});
	placeholder.set(imH);
	O.realize(out_img);

	ofstream fout("dump.tensortxt");
	write_tensor(out_img, fout);
}
