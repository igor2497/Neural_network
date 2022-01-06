#pragma once

#define DEG_TO_RAD					(M_PI / 180.0)
#define RAD_TO_DEG					(180.0 / M_PI)

/*******************************************************************************************************************
 *	   [Y]	 [Z]
 *		A	 7
 *		|   /
 *		|  /
 *		| /
 *		|/
 *		-------> [X]
 ******************************************************************************************************************/

typedef enum euler_ENUM {
	eulerX = 0,
	eulerY,
	eulerZ,
	eulerCount
} euler_E;

class Quaternion {
public:
	float x;
	float y;
	float z;
	float w;

	Quaternion();
	Quaternion(float x, float y, float z, float w);
	float norm();
	void normalize();
	Quaternion conjugate();
	void eulerToQuaternion(float euler[eulerCount]);
	void quaternionToEuler(float euler[eulerCount]);
	Quaternion operator*(Quaternion q);
	void rotationQuaternion(float euler[eulerCount], float angle);
	Quaternion rotate(Quaternion q);
};
