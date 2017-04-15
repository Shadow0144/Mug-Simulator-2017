#include "kernel.hpp"

/// Kernel

Kernel::Kernel()
{

}

Kernel::~Kernel()
{

}

arma::mat Kernel::operator() (const arma::mat X1, const arma::mat X2) const
{
    return kernalize(X1, X2);
}

/// /Kernel
///
/// Compound_Kernel

Compound_Kernel::Compound_Kernel()
{
    _paramCount = 0;
}

Compound_Kernel::Compound_Kernel(const Compound_Kernel& kernel)
{
    /// TODO
}

Compound_Kernel::~Compound_Kernel()
{

}

Compound_Kernel::Ptr Compound_Kernel::New()
{
    Compound_Kernel::Ptr kernel;
    kernel.reset(new Compound_Kernel());
    return kernel;
}

Kernel::Ptr Compound_Kernel::clone()
{
    Compound_Kernel::Ptr kernel;
    kernel.reset(new Compound_Kernel(*this));
    return kernel;
}

void Compound_Kernel::addKernel(Kernel::Ptr kernel)
{
    _kernels.push_back(kernel);
    _paramCount += kernel->numParams();
}

int Compound_Kernel::numParams() const
{
    return _paramCount;
}

double Compound_Kernel::getParam(int index)
{
    int i = 0;
    while (index >= _kernels[i]->numParams())
    {
        index -= _kernels[i++]->numParams();
    }
    double param = _kernels[i]->getParam(index);
    return param;
}

void Compound_Kernel::setParam(int index, double param)
{
    int i = 0;
    while (index >= _kernels[i]->numParams())
    {
        index -= _kernels[i++]->numParams();
    }
    _kernels[i]->setParam(index, param);
}

int Compound_Kernel::numParameters()
{
    return -1;
}

std::string Compound_Kernel::getKernelName()
{
    return "Compound";
}

std::vector<Kernel::Ptr> Compound_Kernel::getKernels() const
{
    return _kernels;
}

void Compound_Kernel::setKernels(const std::vector<Kernel::Ptr>& kernels)
{
    _kernels = kernels;
}

arma::mat Compound_Kernel::gradientP(const arma::mat X, int index) const
{
    int kernel = 0;
    while (index > _kernels[kernel]->numParams())
    {
        index -= _kernels[kernel++]->numParams();
    }
    arma::mat g = _kernels[kernel]->gradientP(X, index);
    return g;
}

arma::mat Compound_Kernel::gradientX(const arma::mat X, int i, int j) const
{
    arma::mat g(X.n_rows, X.n_rows);
    g.zeros();
    for (int i = 0; i < _kernels.size(); i++)
    {
        g += _kernels[i]->gradientX(X, i, j);
    }
    return g;
}

arma::mat Compound_Kernel::kernalize(const arma::mat X1, const arma::mat X2) const
{
    arma::mat K(X1.n_rows, X2.n_rows);
    K.zeros();
    for (int i = 0; i < _kernels.size(); i++)
    {
        K += (*_kernels[i])(X1, X2);
    }
    return K;
}

/// /Compound_Kernel
///
/// RBF_Kernel

RBF_Kernel::RBF_Kernel(double alpha, double beta, double gamma)
{
    _params[0] = alpha;
    _params[1] = beta;
    _params[2] = gamma;
}

RBF_Kernel::RBF_Kernel(const RBF_Kernel& kernel)
{
    _params[0] = kernel._params[0];
    _params[1] = kernel._params[1];
    _params[2] = kernel._params[2];
}

RBF_Kernel::~RBF_Kernel()
{

}

RBF_Kernel::Ptr RBF_Kernel::New(double alpha, double beta, double gamma)
{
    RBF_Kernel::Ptr kernel;
    kernel.reset(new RBF_Kernel(alpha, beta, gamma));
    return kernel;
}

Kernel::Ptr RBF_Kernel::clone()
{
    RBF_Kernel::Ptr kernel;
    kernel.reset(new RBF_Kernel(*this));
    return kernel;
}

int RBF_Kernel::numParams() const
{
    return 3;
}

double RBF_Kernel::getParam(int index)
{
    return _params[index];
}

void RBF_Kernel::setParam(int index, double param)
{
    _params[index] = param;
}

int RBF_Kernel::numParameters()
{
    return 3;
}

std::string RBF_Kernel::getKernelName()
{
    return "RBF";
}

arma::mat RBF_Kernel::gradientP(const arma::mat X, int index) const
{
    arma::mat g(X.n_rows, X.n_rows);
    for (int m = 0; m < X.n_rows; m++)
    {
        for (int n = 0; n < X.n_rows; n++)
        {
            arma::mat XSq = X.row(n) - X.row(m);
            XSq = XSq * XSq.t();
            switch (index)
            {
            case 0:
                g(m, n) = std::exp(-0.5 * _params[2] * XSq(0, 0));
                break;
            case 1:
                g(m, n) = ((m == n) ? 1.0 : 0.0);
                break;
            case 2:
                g(m, n) = -0.5 * _params[0] * XSq(0, 0) * std::exp(-0.5 * _params[2] * XSq(0, 0));
                break;
            }
        }
    }

    return g;
}

arma::mat RBF_Kernel::gradientX(const arma::mat X, int i, int j) const
{
    arma::mat g(X.n_rows, X.n_rows);
    for (int m = 0; m < X.n_rows; m++)
    {
        for (int n = 0; n < X.n_rows; n++)
        {
            if (n == i)
            {
                arma::mat XSq = (X.row(n) - X.row(m));
                XSq = XSq * XSq.t();
                double XSq2 = XSq(0, 0);
                g(m, n) = _params[0] * _params[2] * (X(i, j) - X(m, j)) *
                        std::exp(-0.5 * _params[2] * XSq2);
            }
            else if (m == i)
            {
                arma::mat XSq = (X.row(n) - X.row(m));
                XSq = XSq * XSq.t();
                double XSq2 = XSq(0, 0);
                g(m, n) = -_params[0] * _params[2] * (X(n, j) - X(i, j)) *
                        std::exp(-0.5 * _params[2] * XSq2);
            }
            else
            {
                g(m, n) = 0.0;
            }
        }
    }
    return g;
}

arma::mat RBF_Kernel::kernalize(const arma::mat X1, const arma::mat X2) const
{
    arma::mat K(X1.n_rows, X2.n_rows);
    for (int i = 0; i < X1.n_rows; i++)
    {
        for (int j = 0; j < X2.n_rows; j++)
        {
            arma::mat diff = X1.row(i) - X2.row(j);
            diff = diff * diff.t();
            K(i, j) = _params[0] * std::exp(-0.5 * _params[2] * diff(0, 0)) + ((i == j) ? _params[1] : 0.0);
        }
    }
    return K;
}

/// /RBF_Kernel
///
/// Linear_Kernel

Linear_Kernel::Linear_Kernel(double alpha)
{
    _params[0] = alpha;
}

Linear_Kernel::Linear_Kernel(const Linear_Kernel& kernel)
{
    _params[0] = kernel._params[0];
}

Linear_Kernel::~Linear_Kernel()
{

}

Linear_Kernel::Ptr Linear_Kernel::New(double alpha)
{
    Linear_Kernel::Ptr kernel;
    kernel.reset(new Linear_Kernel(alpha));
    return kernel;
}

Kernel::Ptr Linear_Kernel::clone()
{
    Linear_Kernel::Ptr kernel;
    kernel.reset(new Linear_Kernel(*this));
    return kernel;
}

int Linear_Kernel::numParams() const
{
    return 1;
}

double Linear_Kernel::getParam(int index)
{
    return _params[index];
}

void Linear_Kernel::setParam(int index, double param)
{
    _params[index] = param;
}

int Linear_Kernel::numParameters()
{
    return 1;
}

std::string Linear_Kernel::getKernelName()
{
    return "Linear";
}

arma::mat Linear_Kernel::gradientP(const arma::mat X, int) const
{
    arma::mat g(X.n_rows, X.n_rows);
    for (int i = 0; i < X.n_rows; i++)
    {
        for (int j = 0; j < X.n_rows; j++)
        {
            arma::mat XSq = X.row(j) * X.row(i).t();
            g(i, j) = XSq(0, 0);
        }
    }
    return g;
}

arma::mat Linear_Kernel::gradientX(const arma::mat X, int i, int j) const
{
    arma::mat g = _params[0] * 2 * X;
    return g;
}

arma::mat Linear_Kernel::kernalize(const arma::mat X1, const arma::mat X2) const
{
    arma::mat K = _params[0] * (X1 * X2.t());
    return K;
}

/// /Linear_Kernel
