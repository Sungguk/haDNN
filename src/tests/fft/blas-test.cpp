//File: blas-test.cpp
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <vector>
#include "lib/timer.hh"
#include <unordered_map>
#include <set>
#include <iterator>
#include <unordered_set>
#include <queue>
using namespace std;
#define MSET(ARR, x) memset(ARR, x, sizeof(ARR))
#define REP(x, y) for (auto x = decltype(y){0}; x < (y); x ++)
#define REPL(x, y, z) for (auto x = decltype(z){y}; x < (z); x ++)
#define REPD(x, y, z) for (auto x = decltype(z){y}; x >= (z); x --)
#define P(a) std::cout << (a) << std::endl
#define PP(a) std::cout << #a << ": " << (a) << std::endl
#define PA(arr) \
	do { \
		std::cout << #arr << ": "; \
		std::copy(begin(arr), end(arr), std::ostream_iterator<std::remove_reference<decltype(arr)>::type::value_type>(std::cout, " ")); \
		std::cout << std::endl;  \
	} while (0)
#include <cblas.h>

void simple_test() {
	vector<float> a{1,2,3,4,5,6,7,8};	//2x2

	vector<float> b{1,2,3,4,5,6,7,8};	//2x2
	vector<float> c(10, 0);
	float v1[] = {1, 0};
	float v0[] = {0, 0};
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, v1, a.data(), 2, b.data(), 2, v0, c.data(), 2) ;
	PA(c);
}

void speed_test() {
	int H = 64, W = 64, Cin = 128, Cout = 128, N = 64;
	float* imgfft = new float[H * W * N * Cin * 2];
	float* wfft = new float[H * W * Cin * Cout *  2];
	float* output = new float[H * W * N * Cout * 2];

	float v1[] = {1, 0};
	float v0[] = {0, 0};

	GuardedTimer tm("tm");
	int stride1 = N * Cin * 2;
	int stride2 = Cin * Cout * 2;
	int stride3 = N * Cout * 2;
	REP(i, H) REP(j, W) {
		int stride0 =0;// i * W + j;
		float* A = imgfft + stride0 * stride1;
		float* B = wfft + stride0 * stride2;
		float* C = output + stride0 * stride3;
		cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				N, Cout, Cin, v1,
				A, Cin, B, Cout, v0, C, Cout) ;
		/*
		 *cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		 *    N, Cout, Cin, *v1, A, N, B, Cin, *v0, C, N) ;
		 */
	}

}

int main() {
	//simple_test();
	speed_test();

}
