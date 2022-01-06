#include<stdlib.h>
#include<iostream>
#include<fstream>
#include"nn.h"
#include"quaternion.h"
#include"stick_balancing.h"
#define _USE_MATH_DEFINES
#include<math.h>

#define hidden_layers		2						//!< number of hidden layers in neural network
#define input_layer 		9						//!< number of input values for neural network
#define hidden1 			16						//!< number of nodes in 1st layer
#define hidden2 			16						//!< number of nodes in 2nd layer
#define hidden3 			8						//!< number of nodes in 3rd layer
#define output_layer		4						//!< number of output values
#define population 			256						//!< AI population size
#define mutation 			10						//!< chance per 1000 of random matrix field value 
#define next_gen 			16						//!< number of best cars that will be used to create next generation
#define roulette_size 		2000					//!< size of an array to bias the better cars for selection in new generation

#define PENDULUM_MASS		1						//!< Pendulum mass in kg
#define PENDULUM_HEIGHT		1						//!< Pendulum height in meters
#define PENDULUM_Y_POS		0.5						//!< Pendulum mass center height
#define PENDULUM_X_ANGLE	0.1						//!< Arbitrary angle around X axis
#define PENDULUM_Z_ANGLE	0.1						//!< Arbitrary angle around Z axis

#define TIME_STEP			0.001					//!< Simulation time step in seconds
#define DURATION			8						//!< Simulation duration in seconds
#define ITERATIONS			DURATION / TIME_STEP	//!< Number of iterations in simulation

#define FORCE_CLIPOFF		3

using namespace std;

void main() {
	unsigned int i;
	float axisVector[eulerCount] = { -1, 1, -1 };
	Quaternion pendulumQ(PENDULUM_X_ANGLE, 0, PENDULUM_Z_ANGLE, 1);
	char *unityRotation = "C:/repo/unity/inverted pendulum/Assets/angles.txt";
	char *unityPosition = "C:/repo/unity/inverted pendulum/Assets/position.txt";
	ofstream qFile, pFile;
	double pSize[eulerCount] = { 0.1, PENDULUM_HEIGHT, 0.1 };
	double pPosition[eulerCount] = { 0, PENDULUM_Y_POS, 0 };
	double pVelocity[eulerCount] = { 0, 0, 0 };
	double pAngularV[eulerCount] = { 0, 0, 0 };
	double pForce[eulerCount] = { 0, 0, 0 };

	pendulumQ.normalize();

	Stick pendulum(PENDULUM_MASS,
				   pSize,
				   pPosition,
				   pendulumQ,
				   pVelocity,
				   pAngularV);

	/*Neural_network *nn[population], *temp_nn[population];

	for (i = 0; i < population; i++) {
		nn[i] = new Neural_network(input_layer, hidden1, hidden2, output_layer);
		temp_nn[i] = new Neural_network(input_layer, hidden1, hidden2, output_layer);
	}*/

	// Delete old file and create new
	remove(unityRotation);
	qFile.open(unityRotation);
	remove(unityPosition);
	pFile.open(unityPosition);

	pendulum.position[eulerY] = pendulum.massVector.rotate(pendulum.rotation).y;
	pendulum.position[eulerX] = pendulum.massVector.rotate(pendulum.rotation).x;
	pendulum.position[eulerZ] = pendulum.massVector.rotate(pendulum.rotation).z;

	cout << "Start Simulation" << endl;
	for (i = 0; i < ITERATIONS; i++) {
		pForce[eulerY] = -((pendulum.massVector.conjugate().rotate(pendulum.rotation)).y + pendulum.position[eulerY]) * pendulum.mass / TIME_STEP / TIME_STEP;
		pForce[eulerX] = -((pendulum.massVector.conjugate().rotate(pendulum.rotation)).x + pendulum.position[eulerX]) * pendulum.mass / TIME_STEP / TIME_STEP;
		pForce[eulerZ] = -((pendulum.massVector.conjugate().rotate(pendulum.rotation)).z + pendulum.position[eulerZ]) * pendulum.mass / TIME_STEP / TIME_STEP;

		if (pForce[eulerY] < -pendulum.mass * g * FORCE_CLIPOFF) {
			pForce[eulerY] = -pendulum.mass * g * FORCE_CLIPOFF;
		}
		if (pForce[eulerY] > pendulum.mass * g * FORCE_CLIPOFF) {
			pForce[eulerY] = pendulum.mass * g * FORCE_CLIPOFF;
		}

		if (pForce[eulerX] > pendulum.mass * g * FORCE_CLIPOFF) {
			pForce[eulerX] = pendulum.mass * g * FORCE_CLIPOFF;
		}
		if (pForce[eulerX] < -pendulum.mass * g * FORCE_CLIPOFF) {
			pForce[eulerX] = -pendulum.mass * g * FORCE_CLIPOFF;
		}

		if (pForce[eulerZ] > pendulum.mass * g * FORCE_CLIPOFF) {
			pForce[eulerZ] = pendulum.mass * g * FORCE_CLIPOFF;
		}
		if (pForce[eulerZ] < -pendulum.mass * g * FORCE_CLIPOFF) {
			pForce[eulerZ] = -pendulum.mass * g * FORCE_CLIPOFF;
		}

		pendulum.physics(pForce, TIME_STEP);
		// Write the quaternion parameters to file
		qFile << pendulum.rotation.x;
		qFile << "\n";
		qFile << pendulum.rotation.y;
		qFile << "\n";
		qFile << pendulum.rotation.z;
		qFile << "\n";
		qFile << pendulum.rotation.w;
		qFile << "\n";

		pFile << pendulum.position[eulerX];
		pFile << "\n";
		pFile << pendulum.position[eulerY];
		pFile << "\n";
		pFile << pendulum.position[eulerZ];
		pFile << "\n";
	}

	cout << "Done" << endl;
}
