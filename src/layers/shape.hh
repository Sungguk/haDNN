//File: shape.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once
#include <Halide.h>
#include "lib/debugutils.hh"

namespace hadnn {

class ShapeExpr {
	public:
		ShapeExpr(Halide::Expr a): shape_{a, -1, -1, -1} {}
		ShapeExpr(Halide::Expr a, Halide::Expr b): shape_{a, b, -1, -1} {}
		ShapeExpr(Halide::Expr a, Halide::Expr b, Halide::Expr c): shape_{a, b, c, -1} {}
		ShapeExpr(Halide::Expr a, Halide::Expr b, Halide::Expr c, Halide::Expr d): shape_{a, b, c, d} {}

		Halide::Expr at(int k) const {
			m_assert(k < 4 && k >= 0);
			return shape_[k];
		}

		Halide::Expr operator [](int k) const { return shape_[k]; }
	private:
		Halide::Expr shape_[4];
};

class Shape {
	public:
		Shape(int a): shape_{a, -1, -1, -1} {}
		Shape(int a, int b): shape_{a, b, -1, -1} {}
		Shape(int a, int b, int c): shape_{a, b, c, -1} {}
		Shape(int a, int b, int c, int d): shape_{a, b, c, d} {}

		int at(int k) const {
			m_assert(k < 4 && k >= 0);
			m_assert(shape_[k] != -1);
			return shape_[k];
		}

		int dim() const {
			if (shape_[3] != -1) return 4;
			if (shape_[2] != -1) return 3;
			if (shape_[1] != -1) return 2;
			return 1;
		}

		int operator [](int k) const { return shape_[k]; }
	private:
		int shape_[4];
};

};
