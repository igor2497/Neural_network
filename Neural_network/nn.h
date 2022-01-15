#pragma once
#include"matrix.h"
#include"error.h"

class Neural_network {
public:
	int layers;
	int il;
	int hl1;
	int hl2;
	int hl3;
	int ol;

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
    Neural_network(const char *fileName);
	void clean();
	Matrix calculate(Matrix);
	void randomize(int mutation);
	void copy(Neural_network);
	void save(const char *fileName, ERR_E *err);
	void load(const char *fileName, ERR_E *err);
};
