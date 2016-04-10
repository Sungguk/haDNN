//File: common.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "common.hh"

#include <fstream>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lib/utils.hh"
using namespace Halide;
using namespace cv;

namespace hadnn {

Image<float> random_image(const Shape& shape, string name) {
	int d = shape.dim();
	switch (d) {
		case 1:
			{
				Image<float> ret(shape[0], name);
				REP(i, shape[0]) ret(i) = rand() * 1.0 / RAND_MAX;
				return ret;
			}
		case 2:
			{
				Image<float> ret(shape[0], shape[1], name);
				REP(i, shape[0])
					REP(j, shape[1]) ret(i, j) = rand() * 1.0 / RAND_MAX;
				return ret;
			}
		case 3:
			{
				Image<float> ret(shape[0], shape[1], shape[2], name);
				REP(i, shape[0])
					REP(j, shape[1])
						REP(k, shape[2]) ret(i, j, k) = rand() * 1.0 / RAND_MAX;
				return ret;
			}
		case 4:
			{
				Image<float> ret(shape[0], shape[1], shape[2], shape[3], name);
				REP(i, shape[0])
					REP(j, shape[1])
						REP(k, shape[2])
							REP(l, shape[3]) ret(i, j, k, l) = rand() * 1.0 / RAND_MAX;
				return ret;
			}
		default:
			error_exit("Unsupported Dimension");
	}
}

Halide::Image<float> read_img2d(string fname, int H, int W) {
	Mat im = imread(fname);
	Mat imf; im.convertTo(imf, CV_32FC3);
	Mat imr; cv::resize(imf, imr, cv::Size(W, H));

	Halide::Image<float> ret(W, H, "image");
	REP(i, H) REP(j, W) ret(j, i) = imr.at<cv::Vec3f>(i, j)[0];
	return ret;
}

Halide::Image<float> read_img3d(string fname, int H, int W) {
	Mat im = imread(fname);
	Mat imf; im.convertTo(imf, CV_32FC3);
	Mat imr; cv::resize(imf, imr, cv::Size(W, H));

	Halide::Image<float> ret(W, H, 3, "image");
	REP(i, H) REP(j, W)
		REP(k, 3)
			ret(j, i, k) = imr.at<cv::Vec3f>(i, j)[k];
	return ret;
}

Halide::Image<float> read_img4d_n3hw(
		string fname, int H, int W, int N, int ch) {
	Mat im = imread(fname);
	Mat imf; im.convertTo(imf, CV_32FC3);
	Mat imr; cv::resize(imf, imr, cv::Size(W, H));

	Halide::Image<float> ret(W, H, ch, N, "image");
	REP(i, H) REP(j, W)
		REP(k, ch) REP(t, N)
			ret(j, i, k, t) = imr.at<cv::Vec3f>(i, j)[k%3] / 255.0;
	return ret;
}

void write_tensor(const Image<float>& v, std::ostream& os) {
	os << const_cast<Image<float>&>(v).name();
	for (int d = v.dimensions()-1; d >= 0; d--)
		os << " " << v.extent(d);
	os << endl;
	switch (v.dimensions()) {
		case 1:
			REP(i, v.extent(0))
				os.write((char*)&v(i), sizeof(float));
			break;
		case 2:
			REP(j, v.extent(1))
				REP(i, v.extent(0))
					os.write((char*)&v(i, j), sizeof(float));
			break;
		case 3:
			REP(k, v.extent(2))
				REP(j, v.extent(1))
					REP(i, v.extent(0))
					  os.write((char*)&v(i, j, k), sizeof(float));
			break;
		case 4:
			REP(l, v.extent(3))
				REP(k, v.extent(2))
					REP(j, v.extent(1))
						REP(i, v.extent(0))
					    os.write((char*)&v(i, j, k, l), sizeof(float));
			break;
	}
}

unordered_map<string, Image<float>> read_params(string fname) {
	unordered_map<string, Image<float>> ret;
	ifstream fin(fname);
	while (not fin.eof()) {
		string meta;
		getline(fin, meta);
		if (fin.eof()) break;
		auto param = strsplit(meta, " ");
		string& name = param[0];
		int ndim = param.size() - 1;
		Image<float> im;
		switch (ndim) {
			case 1:
				im = Image<float>(stoi(param[1]), name);
				break;
			case 2:
				im = Image<float>(stoi(param[2]), stoi(param[1]), name);
				break;
			case 3:
				im = Image<float>(stoi(param[3]), stoi(param[2]), stoi(param[1]), name);
				break;
			case 4:
				im = Image<float>(stoi(param[4]), stoi(param[3]), stoi(param[2]), stoi(param[1]), name);
				break;
			default:
				error_exit("Unsupported dim");
		}
		PA(param);
		int nele = 1;
		for (size_t k = 1; k < param.size(); ++k)
			nele *= stoi(param[k]);
		float* ptr = im.data();
		fin.read((char*)ptr, nele * sizeof(float));
		ret[name] = move(im);
	}
	return ret;
}

}
