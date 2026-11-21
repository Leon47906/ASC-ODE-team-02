#include "vector.hpp"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

using namespace ASC_ode;

class MassSpring : public NonlinearFunction {
private:
  double mass;
  double stiffness;

public:
  MassSpring(double m, double k) : mass(m), stiffness(k) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }

  void evaluate(VectorView<double> x, VectorView<double> f) const override {
    f(0) = x(1);
    f(1) = -stiffness / mass * x(0);
  }

  void evaluateDeriv(VectorView<double> x,
                     MatrixView<double> df) const override {
    df = 0.0;
    df(0, 1) = 1;
    df(1, 0) = -stiffness / mass;
  }
};

class RC_Circuit : public NonlinearFunction {
private:
  double resistance;
  double capacitence;
  // U0 is a function from R to R
  std::shared_ptr<NonlinearFunction> U0;

public:
  RC_Circuit(double r, double c, std::shared_ptr<NonlinearFunction> u0)
      : U0(u0), resistance(r), capacitence(c) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }

  void evaluate(VectorView<double> x, VectorView<double> f) const override {
    Vector<> tmp(1);
    U0->evaluate(x.range(1, 2), tmp);
    f(0) = (tmp(0) - x(0)) / (resistance * capacitence);
    f(1) = 1.0;
  }

  void evaluateDeriv(VectorView<double> x,
                     MatrixView<double> df) const override {
    df = 0.0;
    df(0, 0) = -1.0 / (resistance * capacitence);
    Matrix<> tmp(1, 1);
    U0->evaluateDeriv(x.range(1, 2), tmp);
    df(0, 1) = tmp(0, 0) / (resistance * capacitence);
  }
};

int main() {
  double tend = 4 * std::numbers::pi;
  int steps = 100;
  double tau = tend / steps;

  Vector<> y = {1, 0}; // initializer list
  auto rhs = std::make_shared<MassSpring>(1.0, 1.0);

  ExplicitEuler stepper(rhs);
  // ImprovedEuler stepper(rhs);
  // CrankNicolson stepper(rhs);

  std::ofstream outfile("output_test_ode.txt");
  std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++) {
    stepper.doStep(tau, y);

    std::cout << (i + 1) * tau << "  " << y(0) << " " << y(1) << std::endl;
    outfile << (i + 1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }
}
