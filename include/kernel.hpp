#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <armadillo>
#include <memory>

class Kernel
{
public:
    Kernel();
    ~Kernel();

    typedef std::shared_ptr<Kernel> Ptr;
    virtual Ptr clone() = 0;

    arma::mat operator() (const arma::mat X1, const arma::mat X2) const;

    virtual int numParams() const = 0;
    virtual double getParam(int index) = 0;
    virtual void setParam(int index, double param) = 0;

    //virtual static int numParams() const = 0;

    virtual std::string getKernelName() = 0;

    virtual arma::mat gradientP(const arma::mat X, int index) const = 0;
    virtual arma::mat gradientX(const arma::mat X, int i, int j) const = 0;

private:
    virtual arma::mat kernalize(const arma::mat X1, const arma::mat X2) const = 0;
};

class Compound_Kernel : public Kernel
{
public:
    Compound_Kernel();
    Compound_Kernel(const Compound_Kernel& kernel);
    ~Compound_Kernel();

    typedef std::shared_ptr<Compound_Kernel> Ptr;
    static Ptr New();
    Kernel::Ptr clone();

    void addKernel(Kernel::Ptr kernel);

    int numParams() const;
    double getParam(int index);
    void setParam(int index, double param);

    static int numParameters();

    std::string getKernelName();

    std::vector<Kernel::Ptr> getKernels() const;
    void setKernels(const std::vector<Kernel::Ptr>& kernels);

    arma::mat gradientP(const arma::mat X, int index) const;
    arma::mat gradientX(const arma::mat X, int i, int j) const;

private:
    arma::mat kernalize(const arma::mat X1, const arma::mat X2) const;

    std::vector<Kernel::Ptr> _kernels;
    int _paramCount;
};

class RBF_Kernel : public Kernel
{
public:
    RBF_Kernel(double alpha = 1.0, double beta = 0.1, double gamma = 0.1);
    RBF_Kernel(const RBF_Kernel& kernel);
    ~RBF_Kernel();

    typedef std::shared_ptr<RBF_Kernel> Ptr;
    static Ptr New(double alpha = 1.0, double beta = 0.1, double gamma = 0.1);
    Kernel::Ptr clone();

    int numParams() const;
    double getParam(int index);
    void setParam(int index, double param);

    static int numParameters();

    std::string getKernelName();

    arma::mat gradientP(const arma::mat X, int index) const;
    arma::mat gradientX(const arma::mat X, int i, int j) const;

private:
    double _params[3]; // alpha, beta, gamma

    arma::mat kernalize(const arma::mat X1, const arma::mat X2) const;
};

class Linear_Kernel : public Kernel
{
public:
    Linear_Kernel(double alpha = 1.0);
    Linear_Kernel(const Linear_Kernel& kernel);
    ~Linear_Kernel();

    typedef std::shared_ptr<Linear_Kernel> Ptr;
    static Ptr New(double alpha = 1.0);
    Kernel::Ptr clone();

    int numParams() const;
    double getParam(int index);
    void setParam(int index, double param);

    static int numParameters();

    std::string getKernelName();

    arma::mat gradientP(const arma::mat X, int index) const;
    arma::mat gradientX(const arma::mat X, int i, int j) const;

private:
    double _params[1]; // alpha

    arma::mat kernalize(const arma::mat X1, const arma::mat X2) const;
};

#endif // KERNEL_HPP
