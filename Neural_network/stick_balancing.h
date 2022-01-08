#pragma once
#include"quaternion.h"

#define STICK_INERTIA_COEF	12
#define g	9.81	//!< earth gravitational constant

extern euler_E;

class Stick {
public:
	double mass;
	double size[eulerCount];			//!< x,y,z (length, heigth, width)
	double position[eulerCount];
	Quaternion rotation;
	double velocity[eulerCount];
	double angularV[eulerCount];
	Quaternion massVector;
	double inertia;

	Stick(double _mass,
		  double _size[eulerCount],
		  double _position[eulerCount],
		  Quaternion _rotation,
		  double _velocity[eulerCount],
		  double _angular_v[eulerCount]);
	void physics(double force1[eulerCount], double force2[eulerCount], double time_step);
};
