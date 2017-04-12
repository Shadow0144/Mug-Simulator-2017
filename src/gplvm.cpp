#include "gplvm.hpp"

/// LatentCostFunctor

GPLVMCostFunctor::GPLVMCostFunctor(arma::mat& Y, int q, Kernel::Ptr kernel)
{
    _Y = Y;
    _n = Y.n_rows;
    _p = Y.n_cols;
    _q = q;
    _kernel = kernel;
    _ll = -_p*_n/2*std::log(2 * M_PI);
}

GPLVMCostFunctor::~GPLVMCostFunctor()
{

}

bool GPLVMCostFunctor::Evaluate(const double* parameters, double* cost, double* gradient) const
{
    for (int i = 0; i < _kernel->numParams(); i++)
    {
        _kernel->setParam(i, parameters[i]);
    }
    arma::mat X(parameters+_kernel->numParams(), _n, _q);

    arma::mat K = (*_kernel)(X, X);
    arma::mat invK = K.i();

    cost[0] = _ll - _p/2.0*std::log(arma::det(K)) - 0.5*arma::trace(invK*_Y*_Y.t());

    if (gradient != NULL)
    {
        arma::mat alpha = invK * _Y;
        arma::mat dK = (alpha * alpha.t() - _p*invK);

        // Gradient w.r.t. kernel parameters
        int XStart = _kernel->numParams();
        for (int i = 0; i < XStart; i++)
        {
            arma::mat kp = _kernel->gradientP(X, i);
            arma::mat gK = dK * kp;
            gradient[i] = arma::accu(gK);
        }

        // Gradient w.r.t. X
        for (int j = 0; j < _q; j++)
        {
            for (int i = 0; i < _n; i++)
            {
                arma::mat kg = _kernel->gradientX(X, i, j);
                arma::mat gX = dK * kg;
                gradient[XStart + j*_n + i] = arma::accu(gX);
            }
        }
    }
    else { }

    return true;
}

int GPLVMCostFunctor::NumParameters() const
{
    return ((_n * _q) + _kernel->numParams());
}

/// /LatentCostFunctor
///
/// GPLVM

GPLVM::GPLVM(arma::mat& Y, int q, Kernel::Ptr kernel)
{
    _n = Y.n_rows;
    _p = Y.n_cols;
    _q = q;

    _Y = Y;
    _kernel = kernel;
}

GPLVM::GPLVM(const GPLVM& gplvm)
{
    _n = gplvm._n;
    _p = gplvm._p;
    _q = gplvm._q;

    _kernel = gplvm._kernel->clone();
    _Y = gplvm._Y;
    _X = gplvm._X;

    _L = gplvm._L;
    _LY = gplvm._LY;
}

GPLVM::~GPLVM()
{

}

GPLVM::Ptr GPLVM::New(arma::mat& Y, int q, Kernel::Ptr kernel)
{
    GPLVM::Ptr gplvm;
    gplvm.reset(new GPLVM(Y, q, kernel));
    return gplvm;
}

Kernel::Ptr GPLVM::getKernel() const
{
    return _kernel;
}

arma::mat GPLVM::getX() const
{
    return _X;
}

arma::mat GPLVM::getY() const
{
    return _Y;
}

void GPLVM::setKernel(const Kernel::Ptr& kernel)
{
    _kernel = kernel;
}

void GPLVM::setX(const arma::mat& X)
{
    _n = X.n_rows;
    _q = X.n_cols;
    _X = X;

    arma::mat K = (*_kernel)(_X, _X);
    _L = arma::chol(K); /// Plus noise?
    _LY = arma::solve(_L, _Y);
}

void GPLVM::setY(const arma::mat& Y)
{
    _n = Y.n_rows;
    _p = Y.n_cols;
    _Y = Y;

    arma::mat K = (*_kernel)(_X, _X);
    _L = arma::chol(K); /// Plus noise?
    _LY = arma::solve(_L, _Y);
}

int GPLVM::getN() const
{
    return _n;
}

int GPLVM::getP() const
{
    return _p;
}

int GPLVM::getQ() const
{
    return _q;
}

arma::mat GPLVM::predict(const arma::mat& XStar) const
{
    arma::mat KStar = (*_kernel)(_X, XStar);
    arma::mat LStar = arma::solve(_L, KStar);
    arma::mat YStar = LStar.t() * _LY;

    return YStar;
}

double GPLVM::getLogProbability(const arma::mat& XStar) const
{
    arma::mat Lk = arma::solve(_L, (*_kernel)(_X, XStar));
    arma::mat KStar = (*_kernel)(XStar, XStar);
    arma::mat KSum = arma::sum(arma::pow(Lk, 2), 1);
    double s2 = KStar(0, 0) - KSum(0, 0);
    return std::log(std::sqrt(s2));
}

void GPLVM::learn()
{
    learn(_X);
}

void GPLVM::learn(const arma::mat &X)
{
    _X = X;
    arma::mat C = princomp(_Y);
    C = C.head_rows(_q);
    _X = _Y * C.t();

    int kParams = _kernel->numParams();
    int xParams = (_n * _q);
    int totalParams = kParams + xParams;
    double parameters[totalParams];
    for (int i = 0; i < kParams; i++)
    {
        parameters[i] = _kernel->getParam(i);
    }
    for (int i = kParams; i < totalParams; i++)
    {
        parameters[i] = _X.memptr()[i-kParams];
    }

    GPLVMCostFunctor* fof = new GPLVMCostFunctor(_Y, _q, _kernel);
    ceres::GradientProblem problem(fof);

    ceres::GradientProblemSolver::Options options;
    options.max_num_iterations = 25;
    options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
    options.minimizer_progress_to_stdout = true;

    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(options, problem, parameters, &summary);

    std::cout << summary.BriefReport() << "\n";

    for (int i = 0; i < kParams; i++)
    {
        _kernel->setParam(i, parameters[i]);
    }
    for (int i = kParams; i < totalParams; i++)
    {
        _X.memptr()[i-kParams] = parameters[i];
    }

    arma::mat K = (*_kernel)(_X, _X);
    _L = arma::chol(K/* + (arma::eye(_n, _n)*_beta)*/); /// Plus noise?
    _LY = arma::solve(_L, _Y);
}

/// /GPLVM
