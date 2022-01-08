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
void Stick::physics(double force1[eulerCount], double force2[eulerCount], double time_step) {
	unsigned int i;
	float deltaRotation[eulerCount];
	Quaternion rotationQ[eulerCount];
	Quaternion forceQ1(force1[eulerX], force1[eulerY], force1[eulerZ], 0);
	Quaternion forceQ2(force2[eulerX], force2[eulerY], force2[eulerZ], 0);

	// Rotate the force in opposite direction of the object
	forceQ1 = forceQ1.rotate(rotation.conjugate());
	forceQ2 = forceQ2.rotate(rotation.conjugate());

	// Update the object velocity
	velocity[eulerX] += (force1[eulerX] + force2[eulerX]) / mass * time_step;
	velocity[eulerY] += (force1[eulerY] + force2[eulerY] - mass * g) / mass * time_step;
	velocity[eulerZ] += (force1[eulerZ] + force2[eulerZ]) / mass * time_step;

	// Update the object angular velocity (none of the forces have a lever on Y axis, therefore it is not calculated)
	angularV[eulerX] -= (forceQ1.z - forceQ2.z) * massVector.y * time_step / inertia * 2 * M_PI;
	angularV[eulerZ] += (forceQ1.x - forceQ2.x) * massVector.y * time_step / inertia * 2 * M_PI;

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
