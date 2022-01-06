#pragma once
#include<stdlib.h>
#include<time.h>
#include <chrono>
#include<iostream>
#include<math.h>


class Matrix {
public:
	const int row;
	const int col;
	double *matrix;

	Matrix();
	Matrix(int, int, bool);
	void clean();
	double getel(int, int);
	void setel(int, int, double);
	void setmat(Matrix);
	void print();
	void randomize(int mutation);
};

void mul(Matrix, Matrix, Matrix*);
void add(Matrix, Matrix, Matrix*);
void addi(Matrix*, Matrix);
void relu(Matrix*);
void sigm(Matrix*);
