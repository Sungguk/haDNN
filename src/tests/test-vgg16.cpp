//File: test-vgg16.cpp
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include <fstream>
#include <string>
#include "common.hh"
#include "network.hh"
#include "testing.hh"
#include "layers/everything.hh"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace Halide;
using namespace hadnn;
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	m_assert(argc == 3);
	string param_file = argv[1],
				 image_file = argv[2];
	auto params = read_params(param_file);

	ImageParam placeholder(type_of<float>(), 4);
	Input input{placeholder};

	auto conv_param = [&](string name) -> vector<Image<float>>
	{ return {params[name + "/W"], params[name + "/b"]}; };

	Sequential net(input);
	net.add<Conv2D>(conv_param("conv1_1"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Conv2D>(conv_param("conv1_2"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Pooling>(Shape{2,2}, PoolingMode::MAX);
	// 64, 112, 112
	net.add<Conv2D>(conv_param("conv2_1"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Conv2D>(conv_param("conv2_2"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Pooling>(Shape{2,2}, PoolingMode::MAX);
  // 128, 56, 56
	net.add<Conv2D>(conv_param("conv3_1"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Conv2D>(conv_param("conv3_2"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Conv2D>(conv_param("conv3_3"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Pooling>(Shape{2,2}, PoolingMode::MAX);
  // 256, 28, 28
	net.add<Conv2D>(conv_param("conv4_1"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Conv2D>(conv_param("conv4_2"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Conv2D>(conv_param("conv4_3"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Pooling>(Shape{2,2}, PoolingMode::MAX);
  // 512, 14, 14
	net.add<Conv2D>(conv_param("conv5_1"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Conv2D>(conv_param("conv5_2"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Conv2D>(conv_param("conv5_3"), PaddingMode::SAME)
		 .add<ReLU>()
		 .add<Pooling>(Shape{2,2}, PoolingMode::MAX);
  // 512, 7, 7
	net.default_sched();

	auto& O = net.get_output();
	O.print_loop_nest();

	int B = 8;
	Mat im = imread(image_file);
	Mat imf; im.convertTo(imf, CV_32FC3);
	Mat imr; cv::resize(imf, imr, cv::Size(224, 224));
	REP(i, 224) REP(j, 224)
		imr.at<cv::Vec3f>(i, j) -= cv::Vec3f{110,110,110};
	auto imH = mat_to_image(imr, B);
	auto out_img = random_image({B, 512, 7, 7});
	placeholder.set(imH);
	O.realize(out_img);
	ofstream fout("dump.tensortxt");
	write_tensor(out_img, fout);
}
