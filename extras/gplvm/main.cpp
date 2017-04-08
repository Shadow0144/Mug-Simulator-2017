#include <armadillo>
#include <ceres/ceres.h>

#include "gplvm.hpp"

int main(int, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    int n = 5;
    int p = 6;
    int q = 3;

    arma::mat Y = arma::randu(n, p);
#define COMPOUND
#ifdef COMPOUND
    Compound_Kernel::Ptr kernel = Compound_Kernel::New();
    kernel->addKernel(RBF_Kernel::New());
    kernel->addKernel(Linear_Kernel::New());
#else
    Kernel::Ptr kernel = RBF_Kernel::New();
#endif

    GPLVM gplvm(Y, q, kernel);
    gplvm.learn();

    arma::mat X = gplvm.getX();
    arma::mat YDiff = Y - gplvm.predict(X);
    std::cout << "Results:" << std::endl;
    YDiff.print();

    std::cout << "Parameters: " << std::endl;
    for (int i = 0; i < kernel->numParams(); i++)
    {
        std::cout << kernel->getParam(i) << std::endl;
    }

    return 0;
}
