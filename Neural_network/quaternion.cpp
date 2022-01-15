#include"quaternion.h"
#define _USE_MATH_DEFINES
#include<math.h>

#define QUATERNION_ANGLE_DIVIDER	2.0

/*******************************************************************************************************************
 * Default constructor.
 ******************************************************************************************************************/
Quaternion::Quaternion() : x(0), y(0), z(0), w(1) {}

/*******************************************************************************************************************
 * Parametrized constructor.
 ******************************************************************************************************************/
Quaternion::Quaternion(float _x = 0, float _y = 0, float _z = 0, float _w = 1) : x(_x), y(_y), z(_z), w(_w) {}

/*******************************************************************************************************************
 * Function returns the quaternion 'length'.
 ******************************************************************************************************************/
float Quaternion::norm() {
    return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(w, 2));
}

/*******************************************************************************************************************
 * Function normalizes the quaternion to 'length 1'.
 ******************************************************************************************************************/
void Quaternion::normalize() {
    float normal = norm();
    if(normal != 0) {
        x /= normal;
        y /= normal;
        z /= normal;
        w /= normal;
    }
}

/*******************************************************************************************************************
 * Function returns a quaternion with vector part reversed.
 ******************************************************************************************************************/
Quaternion Quaternion::conjugate() {
    return Quaternion(-x, -y, -z, w);
}

/*******************************************************************************************************************
 * Function transforms euler angles to quaternion.
 ******************************************************************************************************************/
void Quaternion::eulerToQuaternion(float euler[eulerCount]) {
    float xRad, yRad, zRad;

    xRad = euler[eulerX] * DEG_TO_RAD / QUATERNION_ANGLE_DIVIDER;
    yRad = euler[eulerY] * DEG_TO_RAD / QUATERNION_ANGLE_DIVIDER;
    zRad = euler[eulerZ] * DEG_TO_RAD / QUATERNION_ANGLE_DIVIDER;

    x = cos(zRad)*cos(yRad)*sin(xRad) - sin(zRad)*sin(yRad)*cos(xRad);
    y = cos(zRad)*sin(yRad)*cos(xRad) + sin(zRad)*cos(yRad)*sin(xRad);
    z = sin(zRad)*cos(yRad)*cos(xRad) - cos(zRad)*sin(yRad)*sin(xRad);
    w = cos(zRad)*cos(yRad)*cos(xRad) + sin(zRad)*sin(yRad)*sin(xRad);
}

/*******************************************************************************************************************
 * Function transforms and returns euler angles from a quaternion.
 * TODO: Needs working, asin function range problem.
 ******************************************************************************************************************/
void Quaternion::quaternionToEuler(float euler[eulerCount]) {
    float test = 2 * (x*z - w*y);
    float xRad, yRad, zRad;

    if(test != 1 && test != -1) {

        xRad = atan2(y*z + w*x, 0.5 - (x*x + y*y));
        yRad = asin(2 * (w*y - x*z));

        if(((floorf(w * 10000) >= floorf((y - x*z) * 10000)) &&
            (floorf(fabs(w) * 10000) <= floorf(fabs(y - x*z) * 10000)) &&
            (y < 0)) ||
            ((floorf(fabs(w) * 10000) >= floorf(fabs(y - x*z) * 10000)) &&
            (y > 0))) {

            yRad *= -1;
        }
        zRad = atan2(x*y + w*z, 0.5 - (y*y + z*z));

    } else if(test == 1) {
        zRad = atan2(x*y + w*z, 0.5 - (y*y + z*z));
        yRad = M_PI / 2.0;
        xRad = -z + atan2(x*y - w*z, x*z + w*y);

    } else if(test == -1) {

        zRad = atan2(x*y + w*z, 0.5 - (y*y + z*z));
        yRad = -M_PI / 2.0;
        xRad = z + atan2(x*y - w*z, x*z + w*y);

    }

    euler[eulerX] = xRad * RAD_TO_DEG;
    euler[eulerY] = yRad * RAD_TO_DEG;
    euler[eulerZ] = zRad * RAD_TO_DEG;
}

/*******************************************************************************************************************
 * Hamilton product quaternion multiplication
 ******************************************************************************************************************/
Quaternion Quaternion::operator*(Quaternion q) {
    float _x, _y, _z, _w;

    _x = x * q.w + w * q.x + y * q.z - z * q.y;
    _y = y * q.w + w * q.y + z * q.x - x * q.z;
    _z = z * q.w + w * q.z + x * q.y - y * q.x;
    _w = w * q.w - x * q.x - y * q.y - z * q.z;

    return Quaternion(_x, _y, _z, _w);
}

/*******************************************************************************************************************
 * Function normalizes the vector to length 1
 ******************************************************************************************************************/
inline void unitVector(float vector[eulerCount]) {
    float length = sqrt(pow(vector[eulerX], 2) + pow(vector[eulerY], 2) + pow(vector[eulerZ], 2));
    if(length != 0) {
        vector[eulerX] /= length;
        vector[eulerY] /= length;
        vector[eulerZ] /= length;
    }
}

/*******************************************************************************************************************
 * Function creates a rotation quaternion. When multiplying another quaternion by this it will rotate it by [angle]
 * in degrees around axis [axisVector].
 ******************************************************************************************************************/
void Quaternion::rotationQuaternion(float axisVector[eulerCount], float angle) {
    // Normalize the axis of rotation vector to length 1
    unitVector(axisVector);

    x = axisVector[eulerX] * sin(angle * DEG_TO_RAD / QUATERNION_ANGLE_DIVIDER);
    y = axisVector[eulerY] * sin(angle * DEG_TO_RAD / QUATERNION_ANGLE_DIVIDER);
    z = axisVector[eulerZ] * sin(angle * DEG_TO_RAD / QUATERNION_ANGLE_DIVIDER);
    w = cos(angle * DEG_TO_RAD / QUATERNION_ANGLE_DIVIDER);
    this->normalize();
}

/*******************************************************************************************************************
* Function rotates the vector by Quaternion [q].
******************************************************************************************************************/
Quaternion Quaternion::rotate(Quaternion q) {
    return q * (*this) * q.conjugate();
}
