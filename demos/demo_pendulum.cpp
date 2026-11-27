#include "autodiff.hpp"
#include "nonlinfunc.hpp"
#include "timestepper.hpp"
#include "vector.hpp"
#include <fstream>
#include <numbers>

using namespace ASC_ode;
using AD = AutoDiff<2>;

class PendulumAD : public NonlinearFunction {
private:
  double m_length;
  double m_gravity;

public:
  PendulumAD(double length, double gravity = 9.81)
      : m_length(length), m_gravity(gravity) {};

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }

  void evaluate(VectorView<double> x, VectorView<double> f) const override {
    T_evaluate<double>(x, f);
  }

  void evaluateDeriv(VectorView<double> x,
                     MatrixView<double> df) const override {
    Vector<AD> x_ad(2);
    Vector<AD> f_ad(2);

    x_ad(0) = Variable<0>(x(0));
    x_ad(1) = Variable<1>(x(1));
    T_evaluate<AD>(x_ad, f_ad);

    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 2; j++) {
        df(i, j) = f_ad(i).deriv()[j];
      }
    }
  }

  template <typename T = double>
  void T_evaluate(VectorView<T> x, VectorView<T> f) const {
    f(0) = x(1);
    f(1) = -m_gravity / m_length * sin(x(0));
  }
};

int main() {
  double tend = 4 * std::numbers::pi;
  int steps = 300;
  double tau = tend / steps;

  Vector<> y = {3 * std::numbers::pi / 4.0, 0}; // initializer list
  auto rhs = std::make_shared<PendulumAD>(1.0);

  // ExplicitEuler stepper(rhs);
  ImprovedEuler stepper(rhs);
  // CrankNicolson stepper(rhs);

  std::ofstream outfile("output_demo_pendulum.txt");
  // std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++) {
    stepper.doStep(tau, y);

    // std::cout << (i + 1) * tau << "  " << y(0) << " " << y(1) << std::endl;
    outfile << (i + 1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }
}
