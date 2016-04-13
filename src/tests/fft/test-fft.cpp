//File: test-fft.cpp
//Author: Yuxin Wu <ppwwyyxx@gmail.com>


#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lib/utils.hh"
#include "lib/timer.hh"
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

template <typename T>
Func make_real_4d(const Image<T> &re) {
	Var x, y, z, w;
	Func ret;
	ret(x, y, z, w) = re(x, y, z, w);
	return ret;
}

// dim0_extent: extent of dim0
Func collect_complex(ComplexFunc in, int dim0_extent) {
	Func ret;
	Var x,y,z,w;
	ret(x, y, z, w) = select(x%2==0, in(x/2,y,z,w).re(), in(x/2, y, z, w).im());
	ret.bound(x, 0, dim0_extent*2).unroll(x, 2);
	return ret;
}

void cgemm_speed_test() {
	int W = 32, H = 32, Cin = 128, N = 64, Cout = 128;
	ComplexFunc imgfft{"img"}, Wfft{"W"};
	Var x{"x"}, y{"y"}, z{"z"}, w{"w"};
	imgfft(x, y, z, w) = ComplexExpr{x + y + z + w, x-y+z-w};
	Wfft(x, y, z, w) = ComplexExpr{x - y - z + w, x+y+z-w};


	ComplexFunc cgemm{"cgemm"};
	RDom rv(0, Cin, "RCin");
	cgemm(x, y, z, w) = ComplexExpr{0,0};
	cgemm(x, y, z, w) += imgfft(x, y, rv.x, w) * Wfft(x, y, z, rv.x);

//	cgemm.bound(x, 0, W).bound(y, 0, H/2+1).bound(z,0,Cin).bound(w,0,N);

	imgfft.compute_root();
	Wfft.compute_root();
	auto&& U = cgemm.update();

	cgemm.reorder_storage(z, w, x, y);
	U.reorder(z, rv.x, w, x, y);
	U.vectorize(z, 8);

	/*
	 *Var zo{"zo"}, wo{"wo"}, zi{"zi"};
	 *RVar rvi{"rvi"}, rvo{"rvo"};
	 *U.split(z, zo, zi, 4);
	 *U.reorder(x, y, zi, rv.x, zo, w);
	 * //U.reorder(x, y, rv.x, z, w);
	 *U.vectorize(x, 8).unroll(x, W / 8);
	 */

	cgemm.compute_root();
	cgemm.print_loop_nest();
	Func cgemm_complex = collect_complex(cgemm, W);
	Image<float> cgemm_out(W * 2, H / 2 + 1, Cout, N);
	cgemm_complex.compile_jit();
	{
		GuardedTimer tm("Cgemm_out_complex");
		cgemm_complex.realize(cgemm_out);
	}
}

Halide::Image<float> run_4d_conv_fft(const Image<float>& img, Image<float>& W) {
	// img: [w,h,c,n]; W: [w,h,o,i]
	Var x{"x"}, y{"y"}, z{"z"}, w{"w"};
	int out_ch = W.extent(2), in_ch = W.extent(3);

	Func padded = BoundaryConditions::constant_exterior(make_real_4d(img), 0,
			{{0, img.extent(0)}, {0, img.extent(1)}, {0, img.extent(2)}, {0, img.extent(3)}});

	Func Wfunc;
	Wfunc(x, y, z, w) = W(W.extent(0) -1 - x, W.extent(1) - 1- y, z, w);
	Func Wpadded = BoundaryConditions::constant_exterior(Wfunc, 0,
			{{0, W.extent(0)}, {0, W.extent(1)}, {0, W.extent(2)}, {0, W.extent(3)}});

	int fftW = img.extent(0) + W.extent(0) / 2,
			fftH = img.extent(1) + W.extent(1) / 2;
	fftW = pow2roundup(fftW);
	fftH = pow2roundup(fftH);
	PP(fftW);

	auto target = get_jit_target_from_environment();
	auto img_fft = fft2d_r2c(padded, fftW, fftH, target);
	auto W_fft = fft2d_r2c(Wpadded, fftW, fftH, target);

	/*
	 *Func img_fft_complex = collect_complex(img_fft, fftW);
	 *img_fft.compute_root();
	 *Image<float> img_fft_out(fftW * 2, fftH / 2 + 1, img.extent(2), img.extent(3), "img_fft_out");
	 *img_fft_complex.compile_jit();
	 *{
	 *  GuardedTimer tm("img_fft_out");
	 *  img_fft_complex.realize(img_fft_out);
	 *}
	 *Image<float> W_fft_out(fftW * 2, fftH / 2 + 1, W.extent(2), W.extent(3), "W_fft_out");
	 *Func W_fft_complex = collect_complex(W_fft, fftW);
	 *W_fft.compute_root();
	 *W_fft_complex.compile_jit();
	 *{
	 *  GuardedTimer tm("W_fft_out");
	 *  W_fft_complex.realize(W_fft_out);
	 *}
	 */

	// img: [w, h/2+1, Cin, N]
	// W: [w, h/2+1, Cout, Cin]
	ComplexFunc cgemm{"cgemm"};
	RDom rv(0, in_ch, "rv");
	cgemm(x, y, z, w) = ComplexExpr{0,0};
	cgemm(x, y, z, w) += img_fft(x, y, rv.x, w) * W_fft(x, y, z, rv.x);

	W_fft.compute_root();
	img_fft.compute_at(cgemm, w);
	//img_fft.compute_root();

	cgemm.compute_root();
	Image<float> cgemm_outr(fftW, fftH/2+1, out_ch, img.extent(3)),
						cgemm_outi(fftW, fftH/2+1, out_ch, img.extent(3));

	cgemm.update().reorder(x, y, rv.x, z, w).vectorize(x, 8);
	/*
	 *cgemm.compute_root();
	 *cgemm.compile_jit();
	 *{
	 *  GuardedTimer tm("cgemm_out");
	 *  cgemm.realize({cgemm_outr, cgemm_outi});
	 *}
	 */

	Fft2dDesc desc; desc.gain = 1.0f / (fftW * fftH);
	Func ifft = fft2d_c2r(cgemm, fftW, fftH, target, desc);
	Func output;
	output(x, y, z, w) = ifft(x + W.extent(0)/2, y + W.extent(1)/2, z, w);

	cgemm.compute_at(ifft, w);//.update().reorder(x, y, rv.x, z, w).vectorize(x, 8);
	ifft.compute_at(output, w);


	Image<float> ifft_out(img.extent(0), img.extent(1), out_ch, img.extent(3));
	output.compile_jit(target);
	GuardedTimer tm("ifft");
	output.realize(ifft_out);
	return ifft_out;
}

// tested
Halide::Image<float> run_conv_fft(const Image<float>& img, Image<float>& W) {
	Var x{"x"}, y{"y"};
	Func imgfunc = make_real(img);
	Func padded = BoundaryConditions::constant_exterior(imgfunc, 0,
			{{0,img.extent(0)}, {0, img.extent(1)}});
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

Image<float> run_4d_conv_hwcn(const Image<float>& img_old, const Image<float>& W) {
	int out_ch = W.extent(2), in_ch = W.extent(3);

	Image<float> img = random_image(
			{img_old.extent(3), in_ch,
			img_old.extent(0), img_old.extent(1)});

	Image<float> b(out_ch); REP(i, out_ch) b(i) = 0;

	ImageParam placeholder(type_of<float>(), 4);
	Input input{placeholder};
	Conv2DHWCN conv(&input, {W, b}, PaddingMode::SAME);
	conv.default_sched();
	auto& O = conv.get_output();
	placeholder.set(img);
	Image<float> ret(img.extent(0), out_ch, img.extent(2), img.extent(3));
	O.compile_jit();
	{
		GuardedTimer tm("4d conv");
		O.realize(ret);
	}
	return ret;
}

// NCHW
Image<float> run_4d_conv_nchw(const Image<float>& img, const Image<float>& W) {
	int out_ch = W.extent(2);

	Image<float> b(out_ch); REP(i, out_ch) b(i) = 0;

	ImageParam placeholder(type_of<float>(), 4);
	Input input{placeholder};
	Conv2DNCHW conv(&input, {W, b}, PaddingMode::SAME);
	conv.default_sched();
	auto& O = conv.get_output();
	placeholder.set(img);
	Image<float> ret(img.extent(0), img.extent(1), out_ch, img.extent(3));
	O.compile_jit();
	{
		GuardedTimer tm("4d conv");
		O.realize(ret);
	}
	return ret;
}

void test_2d() {
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
}

void test_4d() {
	ImageParam placeholder(type_of<float>(), 4);
	int B = 64, H = 15, W = 15;
	int in_ch = 128, out_ch = 128;
	Halide::Image<float> input_img = read_img4d_n3hw("/home/wyx/proj/cat.png", H, W, B, in_ch);
	Halide::Image<float> Weight = random_image({3,3, out_ch, in_ch}, "Weight");

	auto conv_out = run_4d_conv_hwcn(input_img, Weight);
	auto fft_out = run_4d_conv_fft(input_img, Weight);

	ofstream fout("param.tensortxt");
	write_tensor(input_img, fout);
	write_tensor(Weight, fout);
	write_tensor(conv_out, fout);
	write_tensor(fft_out, fout);
	fout.close();
}

void full_example() {
	auto target = get_jit_target_from_environment();
	int N = 64, Cin = 128, Cout = 128, H = 64, W = 64;
	Var x, y, z, n;
	Image<float> img(W, H, Cin, N), kernel(W, H, Cin, Cout);
	ComplexFunc dft_img = fft2d_r2c(make_real_4d(img), W, H, target);
	ComplexFunc dft_kernel = fft2d_r2c(make_real_4d(kernel), W, H, target);

	ComplexFunc cgemm{"cgemm"};
	RDom rv(0, Cin, "rv");
	cgemm(x, y, z, n) = ComplexExpr{0,0};
	cgemm(x, y, z, n) += dft_img(x, y, rv.x, n) * dft_kernel(x, y, rv.x, z);
	cgemm.bound(x, 0, W).bound(y, 0, H/2+1).bound(z,0,Cout).bound(n,0,N);
	//cgemm(x, y, z, n) = sum(dft_img(x, y, rv.x, n) * dft_kernel(x, y, rv.x, z));
	Func result = fft2d_c2r(cgemm, W, H, target);

	//dft_img.compute_at(cgemm, n);
	dft_kernel.compute_root();
	dft_img.compute_root();
	cgemm.compute_at(result, n).update().reorder(x, y, rv.x, z, n).vectorize(x, 8);
	//cgemm.compute_root().update().reorder(x, y, rv.x, z, n);
	/*
	 *dft_img.compute_at(cgemm, rv.x);
	 *dft_kernel.compute_root();
	 *cgemm.compute_at(result, n);
	 *cgemm.update(0).reorder(x, y, z, rv.x, n).vectorize(x, target.natural_vector_size<float>());
	 */


	result.compute_root();

	Image<float> output(W, H, Cout, N);
	result.compile_jit();
	GuardedTimer tm("output");
	result.realize(output);
}

int main() {
	//test_2d();
	test_4d();
	//cgemm_speed_test();
	//full_example();
}
