#pragma once
#include"matrix.h"

class Neural_network {
public:
	const int layers;
	const int il;
	const int hl1;
	const int hl2;
	const int hl3;
	const int ol;

	Matrix *ih1;
	Matrix *h12;
	Matrix *h23;
	Matrix *ho;

	Matrix *r1;
	Matrix *r2;
	Matrix *r3;

	Matrix *b1;
	Matrix *b2;
	Matrix *b3;
	Matrix *ob;
	Matrix *outputs;

	Neural_network();
	Neural_network(int, int, int, int, int);
	Neural_network(int, int, int, int);
	Neural_network(int, int, int);
	void clean();
	Matrix calculate(Matrix);
	void randomize(int mutation);
	void copy(Neural_network);
};