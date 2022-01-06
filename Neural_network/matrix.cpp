#include "matrix.h"

Matrix::Matrix():row(1),col(1) {
	matrix = (double*)malloc(sizeof(double));
}

Matrix::Matrix(int _row, int _col, bool _randomize)
	: row(_row), col(_col)
{
	int i;
	matrix = (double*)malloc(sizeof(double) * _row * _col);
	if (_randomize) {
		if (matrix) {
			for (i = 0; i < _row * _col; i++) {
				*(matrix + i) = (double)((double)rand() / RAND_MAX * 2 - 1);
			}
		}
		else {
			std::cout << "NULL matrix malloc" << std::endl;
		}
	}
}

double Matrix::getel(int i, int j) {
	return *(matrix + i*col + j);
}

void Matrix::setel(int i, int j, double x) {
	*(matrix + i*col + j) = x;
}

void Matrix::setmat(Matrix a) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			*(matrix + i*col + j) = a.getel(i, j);
		}
	}
}

void Matrix::print() {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			std::cout << getel(i, j) << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void mul(Matrix mat1, Matrix mat2, Matrix *res) {
	int i, j, k;
	for (i = 0; i < res->row; i++) {
		for (j = 0; j < res->col; j++) {
			res->setel(i, j, 0);
		}
	}
	if (mat1.col == mat2.row) {
		for (i = 0; i < mat1.row; i++) {
			for (j = 0; j < mat2.col; j++) {
				for (k = 0; k < mat1.col; k++) {
					res->setel(i,j, res->getel(i,j) + mat1.getel(i,k)*mat2.getel(k,j));
				}
			}
		}
	}
}

void add(Matrix mat1, Matrix mat2, Matrix *res) {
	if (mat1.row == mat2.row && mat1.col == mat2.col) {
		for (int i = 0; i < mat1.row; i++) {
			for (int j = 0; j < mat1.col; j++) {
					res->setel(i, j, mat1.getel(i, j) + mat2.getel(i, j));
			}
		}
	}
}

void addi(Matrix *mat1, Matrix mat2) {
	if (mat1->row == mat2.row && mat1->col == mat2.col) {
		for (int i = 0; i < mat1->row; i++) {
			for (int j = 0; j < mat1->col; j++) {
				mat1->setel(i, j, mat1->getel(i, j) + mat2.getel(i, j));
			}
		}
	}
}


void relu(Matrix *a) {
	for (int i = 0; i < a->row*a->col; i++) {
		if (a->matrix[i] < 0) {
			a->matrix[i] = 0;
		}
	}
}

void sigm(Matrix *a) {
	for (int i = 0; i < a->row*a->col; i++) {
		a->matrix[i] = 1 / (1 + exp(-a->matrix[i]));
	}
}

void Matrix::randomize(int mutation) {
	for (int i = 0; i < row*col; i++) {
		if (rand() % 1000 < mutation) {
			matrix[i] = (double)rand() / RAND_MAX * 2 - 1;
		}
	}
}

void Matrix::clean() {
	free(matrix);
}
