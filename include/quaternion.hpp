#ifndef QUATERNION_HPP
#define QUATERNION_HPP

#include <armadillo>

class Quaternion
{
public:
    Quaternion();
    Quaternion(double x, double y, double z, double w);
    Quaternion(arma::vec xyz, double w);
    Quaternion(arma::vec baseVector, arma::vec vector);
    ~Quaternion();

    double getX() const;
    double getY() const;
    double getZ() const;
    double getW() const;
    arma::vec getXYZ() const;

    void setX(double x);
    void setY(double y);
    void setZ(double z);
    void setW(double w);
    void setXYZ(arma::vec xyz);

    void normalize();

    arma::vec transformPoint(const arma::vec& point) const;
    arma::mat transformPoints(const arma::mat& points) const;

    Quaternion operator+(const Quaternion& q) const;
    Quaternion operator*(const Quaternion& q) const;

private:
    double _x;
    double _y;
    double _z;
    double _w;
};

#endif // QUATERNION_HPP
