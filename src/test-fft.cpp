//File: test-fft.cpp
//Author: Yuxin Wu <ppwwyyxx@gmail.com>


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lib/utils.hh"
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

	Halide::Image<float> ret(H, W);
	REP(i, H) REP(j, W) ret(i, j) = imr.at<cv::Vec3f>(i, j)[0];
	return ret;
}

int main() {
	ImageParam placeholder(type_of<float>(), 2);
	int H = 32, W = 32;

	Input input{placeholder};
	auto target = get_jit_target_from_environment();

	Halide::Image<float> tmp = read_img2d(
			"/home/wyx/proj/cat.png", H, W);
	auto in = make_real(tmp);
	PP(tmp(20, 20));

	/*
	 *auto padded = BoundaryConditions::constant_exterior(
	 *    input.get_output(), 0, {{0, H}, {0, W}});
	 */

	Fft2dDesc desc;
	ComplexFunc fft_im = fft2d_r2c(in, H, W, target);
	///Func fft_inv = fft2d_c2r(fft_im, H, W, get_jit_target_from_environment(), desc);
	Func fft_inv = fft2d_c2r(fft_im, H, W, target);

	/*
	 *Halide::Image<float> input_img = read_img2d(
	 *    "/home/wyx/proj/cat.png", H, W);
	 *placeholder.set(input_img);
	 */


	fft_im.compute_root();
	Var x, y;
	/*
	 *Func out_re;
	 *out_re(x, y) = re(fft_im(x, y));
	 */

	Halide::Image<float> output = fft_inv.realize(H, W, target);
	PP(output(20,20) * 1.0 / H / W);
}
