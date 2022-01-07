#include"stick_balancing.h"
#define _USE_MATH_DEFINES
#include<math.h>

/*******************************************************************************************************************
 * Parametrized constructor.
 ******************************************************************************************************************/
Stick::Stick(double _mass,
			 double _size[eulerCount],
			 double _position[eulerCount],
			 Quaternion _rotation,
			 double _velocity[eulerCount],
			 double _angular_v[eulerCount])
			 : mass(_mass), rotation(_rotation) {

	unsigned int i;
	for (i = 0u; i < eulerCount; i++) {
		size[i] = _size[i];
		position[i] = _position[i];
		velocity[i] = _velocity[i];
		angularV[i] = _angular_v[i];
	}

	massVector.x = 0;
	massVector.y = size[eulerY] / 2;
	massVector.z = 0;
	massVector.w = 0;

	inertia = mass * size[eulerY] * size[eulerY] / STICK_INERTIA_COEF;
}

/*******************************************************************************************************************
 * Function takes in the forces applied to the object and time step, and calculates the new object position and
 * and rotation.
 ******************************************************************************************************************/
void Stick::physics(double force[eulerCount], double time_step) {
	unsigned int i;
	float deltaRotation[eulerCount];
	Quaternion rotationQ[eulerCount];
	Quaternion forceQ(force[eulerX], force[eulerY], force[eulerZ], 0);

	// Rotate the force in opposite diraction of the object
	forceQ = forceQ.rotate(rotation.conjugate());

	// Update the object velocity
	velocity[eulerX] += force[eulerX] / mass * time_step;
	velocity[eulerY] += (force[eulerY] - mass * g) / mass * time_step;
	velocity[eulerZ] += force[eulerZ] / mass * time_step;

	// Update the object angular velocity (none of the forces have a lever on Y axis, therefore it is not calculated)
	angularV[eulerX] -= forceQ.z * massVector.y * time_step / inertia * 2 * M_PI;
	angularV[eulerZ] += forceQ.x * massVector.y * time_step / inertia * 2 * M_PI;

	// Update position and rotation quaternion for every axis
	for (i = 0u; i < eulerCount; i++) {
		position[i] += velocity[i] * time_step;
		deltaRotation[eulerX] = deltaRotation[eulerY] = deltaRotation[eulerZ] = 0;
		deltaRotation[i] = 1;
		rotationQ[i].rotationQuaternion(deltaRotation, angularV[i] * time_step);
	}

	// Update the object orientation
	rotation = rotation * rotationQ[eulerX] * rotationQ[eulerY] * rotationQ[eulerZ];
	rotation.normalize();
}
