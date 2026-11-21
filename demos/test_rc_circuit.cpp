#include "vector.hpp"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

using namespace ASC_ode;

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
  double tend = 0.5;
  int steps = 400;
  double tau = tend / steps;
  double resistance = 100.0, capacitence = 1e-6;

  Vector<> y = {0, 0}; // initializer list
  auto c = std::make_shared<Cosine>(1);
  auto id = std::make_shared<IdentityFunction>(1);
  auto phase_param = std::make_shared<Parameter>(100 * std::numbers::pi);
  auto scaled = std::make_shared<ScaleFunction>(id, phase_param);
  auto U0 = Compose(c, scaled);
  auto rhs_rc = std::make_shared<RC_Circuit>(resistance, capacitence, U0);

  // ExplicitEuler stepper_rc(rhs_rc);
  ImplicitEuler stepper_rc(rhs_rc);
  // CrankNicolson stepper_rc(rhs_rc);
  std::ofstream outfile("output_test_ode_rc.txt");
  std::cout << 0.0 << "  " << y(0) << std::endl;
  outfile << 0.0 << "  " << y(0) << std::endl;

  for (int i = 0; i < steps; i++) {
    stepper_rc.doStep(tau, y);

    std::cout << (i + 1) * tau << "  " << y(0) << std::endl;
    outfile << (i + 1) * tau << "  " << y(0) << std::endl;
  }
}
