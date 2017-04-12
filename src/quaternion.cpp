#include <quaternion.hpp>

Quaternion::Quaternion()
{
    _x = 0;
    _y = 0;
    _z = 0;
    _w = 1;
}

Quaternion::Quaternion(double x, double y, double z, double w)
{
    _x = x;
    _y = y;
    _z = z;
    _w = w;
}

Quaternion::Quaternion(arma::vec xyz, double w)
{
    _x = xyz(0);
    _y = xyz(1);
    _z = xyz(2);
    _w = w;
}

Quaternion::Quaternion(arma::vec baseVector, arma::vec vector)
{
    _w = arma::dot(baseVector, vector);
    arma::vec xyz = arma::cross(baseVector, vector);
    _x = xyz(0);
    _y = xyz(1);
    _z = xyz(2);
    _w += 1;
    normalize();
}

Quaternion::~Quaternion()
{

}

double Quaternion::getX() const
{
    return _x;
}
double Quaternion::getY() const
{
    return _y;
}

double Quaternion::getZ() const
{
    return _z;
}
double Quaternion::getW() const
{
    return _w;
}

arma::vec Quaternion::getXYZ() const
{
    arma::vec xyz(3);
    xyz(0) = _x;
    xyz(1) = _y;
    xyz(2) = _z;
    return xyz;
}

void Quaternion::setX(double x)
{
    _x = x;
}

void Quaternion::setY(double y)
{
    _y = y;
}

void Quaternion::setZ(double z)
{
    _z = z;
}

void Quaternion::setW(double w)
{
    _w = w;
}

void Quaternion::setXYZ(arma::vec xyz)
{
    _x = xyz(0);
    _y = xyz(1);
    _z = xyz(2);
}

void Quaternion::normalize()
{
    double n = 1.0/std::sqrt(_x*_x + _y*_y + _z*_z + _w*_w);
    _x *= n;
    _y *= n;
    _z *= n;
    _w *= n;
}

arma::vec Quaternion::transformPoint(const arma::vec& point) const
{
    arma::vec pPrime = point.head(3);
    Quaternion p(pPrime, 1);
    Quaternion q(_x, _y, _z, _w);
    Quaternion qStar(-_x, -_y, -_z, _w);
    p = q * p * qStar;
    pPrime = p.getXYZ() * p.getW();
    return pPrime;
}

arma::mat Quaternion::transformPoints(const arma::mat& points) const
{
    arma::mat pPrimes(points.n_rows, points.n_cols);
    for (int i = 0; i < points.n_rows; i++)
    {
        arma::vec pVec = points.row(i);
        pVec = pVec.cols(0, 2);
        Quaternion p(pVec, 1);
        Quaternion q(_x, _y, _z, _w);
        Quaternion qStar(-_x, -_y, -_z, _w);
        p = q * p * qStar;
        arma::vec pPrime = p.getXYZ() * p.getW();
        pPrimes.row(i) = pPrime;
    }
    return pPrimes;
}

Quaternion Quaternion::operator+(const Quaternion& q) const
{
    return Quaternion(_x+q._x, _y+q._y, _z+q._z, _w+q._w);
}

Quaternion Quaternion::operator*(const Quaternion& q) const
{
    arma::vec xyz = _w*q.getXYZ() + q._w*getXYZ() + (arma::cross(getXYZ(), q.getXYZ()));
    double w = _w*q._w - arma::dot(getXYZ(), q.getXYZ());
    return Quaternion(xyz, w);
}
