#include<stdlib.h>
#include<iostream>
#include"nn.h"

#define hidden_layers	2			//!< number of hidden layers in neural network
#define input_layer 	9			//!< number of input values for neural network
#define hidden1 		16			//!< number of nodes in 1st layer
#define hidden2 		16			//!< number of nodes in 2nd layer
#define hidden3 		8			//!< number of nodes in 3rd layer
#define output_layer	4			//!< number of output values
#define population 		256			//!< AI population size
#define mutation 		10			//!< chance per 1000 of random matrix field value 
#define next_gen 		16			//!< number of best cars that will be used to create next generation
#define roulette_size 	2000		//!< size of an array to bias the better cars for selection in new generation

using namespace std;

void main() {
	int i;
	double number = 24.42;
	Neural_network *nn[population], *temp_nn[population];

	for (i = 0; i < population; i++) {
		nn[i] = new Neural_network(input_layer, hidden1, hidden2, output_layer);
		temp_nn[i] = new Neural_network(input_layer, hidden1, hidden2, output_layer);
	}
	cout << "Number: " << number << endl;
	cin >> number;
}