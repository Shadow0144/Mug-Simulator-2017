#ifndef GPLVM_HPP
#define GPLVM_HPP

#include <armadillo>
#include <ceres/ceres.h>

#include "kernel.hpp"

class GPLVMCostFunctor : public ceres::FirstOrderFunction
{
public:
    GPLVMCostFunctor(arma::mat Y, int q, Kernel::Ptr kernel);

    ~GPLVMCostFunctor();

    bool Evaluate(const double* parameters, double* cost, double* gradient) const;

    int NumParameters() const;

private:
    arma::mat _Y;
    int _n;
    int _p;
    int _q;
    double _ll;
    Kernel::Ptr _kernel;
};

class GPLVM
{
public:
    GPLVM(arma::mat Y, int q, Kernel::Ptr kernel);
    GPLVM(const GPLVM& gplvm);
    ~GPLVM();

    typedef std::shared_ptr<GPLVM> Ptr;
    static Ptr New(arma::mat Y, int q, Kernel::Ptr kernel);

    Kernel::Ptr getKernel();
    arma::mat getX();
    arma::mat getY();

    arma::mat predict(arma::mat XStar);

    void learn();

private:
    int _n;
    int _p;
    int _q;

    Kernel::Ptr _kernel;
    arma::mat _Y;
    arma::mat _X;

    arma::mat _L;
    arma::mat _LY;
};

#endif
