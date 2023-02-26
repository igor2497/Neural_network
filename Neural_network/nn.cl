__kernel void nn_matrix(__constant const double *inputs, __constant const double *matrix, __constant const double *bias, __constant const int *dimensions, __global double *output) {
 
    // Get the index of the current element to be processed
    const int id = get_global_id(0);
	int columns = dimensions[0];
    int i;
	double temp = 0;

    for(i = 0; i < columns; i++) {
		temp += inputs[i] * matrix[id * columns + i];
    }

	temp += bias[id];
	
    temp = 1 / (1 + exp(-temp));

	output[id] = temp;
}