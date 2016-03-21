//File: common.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "common.hh"

#include <fstream>
#include <sstream>

#include "lib/utils.hh"
using namespace Halide;

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
		auto ss = strsplit(meta, " ");
		string& name = ss[0];
		int ndim = ss.size() - 1;
		Image<float> im;
		switch (ndim) {
			case 1:
				im = Image<float>(stoi(ss[1]), name);
				break;
			case 2:
				im = Image<float>(stoi(ss[2]), stoi(ss[1]), name);
				break;
			case 3:
				im = Image<float>(stoi(ss[3]), stoi(ss[2]), stoi(ss[1]), name);
				break;
			case 4:
				im = Image<float>(stoi(ss[4]), stoi(ss[3]), stoi(ss[2]), stoi(ss[1]), name);
				break;
			default:
				error_exit("Unsupported dim");
		}
		int nele = 1;
		for (size_t k = 1; k < ss.size(); ++k)
			nele *= stoi(ss[k]);
		float* ptr = im.data();
		fin.read((char*)ptr, nele * sizeof(float));
		ret[name] = move(im);
	}
	return ret;
}

}
