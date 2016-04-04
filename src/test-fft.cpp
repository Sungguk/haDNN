//File: test-fft.cpp
//Author: Yuxin Wu <ppwwyyxx@gmail.com>


#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lib/utils.hh"
#include "common.hh"
#include "layers/fft/fft.h"
#include "layers/everything.hh"

using namespace cv;
using namespace hadnn;
using namespace Halide;



template <typename T>
Func make_real(const Image<T> &re) {
	Var x, y;
	Func ret;
	ret(x, y) = re(x, y);
	return ret;
}

Halide::Image<float> read_img2d(string fname, int H, int W) {
	Mat im = imread(fname);
	Mat imf; im.convertTo(imf, CV_32FC3);
	Mat imr; cv::resize(imf, imr, cv::Size(W, H));

	Halide::Image<float> ret(W, H, "image");
	REP(i, H) REP(j, W) ret(j, i) = imr.at<cv::Vec3f>(i, j)[0];
	return ret;
}

// tested
Halide::Image<float> run_conv_fft(const Image<float>& img, Image<float>& W) {
	Var x{"x"}, y{"y"};
	Func imgfunc = make_real(img);
	Func padded = BoundaryConditions::constant_exterior(imgfunc, 0, {{0,img.extent(0)}, {0, img.extent(1)}});
	Func Wfunc;
	Wfunc(x, y) = W(W.extent(0) -1 - x, W.extent(1) - 1- y);
	Func Wpadded = BoundaryConditions::constant_exterior(Wfunc, 0, {{0, W.extent(0)}, {0, W.extent(1)}});

	int fftW = img.extent(0) + W.extent(0) / 2,
			fftH = img.extent(1) + W.extent(1) / 2;
	fftW = pow2roundup(fftW);
	fftH = pow2roundup(fftH);
	PP(fftW);

	auto target = get_jit_target_from_environment();
	auto img_fft = fft2d_r2c(padded, fftW, fftH, target);
	auto W_fft = fft2d_r2c(Wpadded, fftW, fftH, target);

	W_fft.compute_root();
	img_fft.compute_root();
	ComplexFunc mult; mult(x, y) = img_fft(x, y) * W_fft(x, y);
	mult.compute_root();

	Fft2dDesc desc;
	desc.gain = 1.0f / (fftW * fftH);
	auto c2r = fft2d_c2r(mult, fftW, fftH, target, desc);
	c2r.compute_root();
	Func output;
	output(x, y) = c2r(x + W.extent(0)/2, y + W.extent(1)/2);
	output.compile_jit();

	Image<float> ret(img.extent(0), img.extent(1), "outputFFT");
	output.realize(ret);
	return ret;
}

// 2d -> 2d conv. tested
Image<float> run_conv(const Image<float>& img, const Image<float>& W) {
	Func padded = BoundaryConditions::constant_exterior(img, 0);
	RDom kernel{0, W.extent(0), 0, W.extent(1)};
	Func output;
	Var x{"x"}, y{"y"};
	output(x, y) = Halide::sum(W(kernel.x, kernel.y) * padded(x + kernel.x - W.extent(0)/2,
				y + kernel.y - W.extent(1)/2));
	output.compile_jit();

	Image<float> ret(img.extent(0), img.extent(1), "outputConv");
	output.realize(ret);
	return ret;
}

int main() {
	ImageParam placeholder(type_of<float>(), 2);
	int H = 8, W = 8;
	Halide::Image<float> input_img = read_img2d("/home/wyx/proj/cat.png", H, W);
	Halide::Image<float> Weight = random_image({3,3}, "Weight");
	auto output_conv = run_conv(input_img, Weight);
	auto output_fft = run_conv_fft(input_img, Weight);
	ofstream fout("param.tensortxt");
	write_tensor(input_img, fout);
	write_tensor(Weight, fout);
	write_tensor(output_conv, fout);
	write_tensor(output_fft, fout);
	fout.close();

	Input input{placeholder};

}
