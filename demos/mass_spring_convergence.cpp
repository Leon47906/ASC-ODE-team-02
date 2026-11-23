#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <nonlinfunc.hpp>
#include <numbers>
#include <timestepper.hpp>
#include <vector>

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

double observed_order(const std::vector<double> &e) {
  size_t n = e.size();
  double p = 0;
  for (size_t i = 0; i < n - 1; i++) {
    p += std::log2(e[i] / e[i + 1]);
  }
  return p / (n - 1);
}

using Factory = std::function<std::unique_ptr<TimeStepper>(
    std::shared_ptr<NonlinearFunction>)>;

struct Case {
  const char *name;
  Factory make;
  double order;
};

double final_error(const Factory &make, int N, double T) {
  auto rhs = std::make_shared<MassSpring>(1.0, 1.0);
  auto stepper = make(rhs);
  Vector<> y = {1, 0}; // initializer list
  double tau = T / N;
  for (int i = 0; i < N; i++)
    stepper->doStep(tau, y);
  return std::max(std::abs(y(0) - 1.0), std::abs(y(1)));
}

using Catch::Matchers::WithinAbs;

TEST_CASE("Convergence order of one-step methods", "[convergence]") {
  std::vector<int> Ns(10);
  for (size_t i = 0; i < Ns.size(); i++) {
    Ns[i] = std::pow(2, i + 4);
  }

  auto c = GENERATE(
      Case{"ExplicitEuler",
           [](auto rhs) { return std::make_unique<ExplicitEuler>(rhs); }, 1.0},
      Case{"ImplicitEuler",
           [](auto rhs) { return std::make_unique<ImplicitEuler>(rhs); }, 1.0},
      Case{"CrankNicolson",
           [](auto rhs) { return std::make_unique<CrankNicolson>(rhs); }, 2.0});

  double tend = 2 * std::numbers::pi;
  std::vector<double> errs;
  for (int N : Ns) {
    errs.push_back(final_error(c.make, N, tend));
  }
  INFO("method: " << c.name);
  REQUIRE_THAT(observed_order(errs), WithinAbs(c.order, 0.1));
}
