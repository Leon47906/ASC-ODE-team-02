// Catch2 tests for autodiff.hpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "autodiff.hpp"

using namespace ASC_ode;
using Catch::Approx;

namespace {

using AD1 = AutoDiff<2>;      // value + gradient
using AD2 = AutoDiff<2, AD1>; // nested: value + gradient + Hessian

Approx approx(double v) { return Approx(v).epsilon(1e-12).margin(1e-12); }

// check value and both partial derivatives of a first-order AutoDiff
void check(const AD1 &f, double val, double d0, double d1) {
  CHECK(f.value() == approx(val));
  CHECK(f.deriv()[0] == approx(d0));
  CHECK(f.deriv()[1] == approx(d1));
}

// nested seed variables: x = variable 0, y = variable 1 at both levels
struct Vars2 {
  AD2 x, y;
};

Vars2 make_nested(double u, double v) {
  AD1 iu = Variable<0>(u);
  AD1 iv = Variable<1>(v);
  return {AD2(Variable<0, AD1>(iu)), AD2(Variable<1, AD1>(iv))};
}

// generic function, works for double and for AutoDiff types
template <typename T> T func(const T &x, const T &y) {
  return sin(x * y) * exp(x) / (y + x);
}

} // namespace

TEST_CASE("Construction", "[autodiff]") {
  SECTION("default constructor is zero") {
    AD1 a;
    check(a, 0, 0, 0);
  }

  SECTION("constant has zero gradient") {
    AD1 c(3.0);
    check(c, 3.0, 0, 0);
  }

  SECTION("Variable seeds a unit vector") {
    AD1 x = Variable<0>(2.0);
    AD1 y = Variable<1>(5.0);
    check(x, 2.0, 1, 0);
    check(y, 5.0, 0, 1);
  }

  SECTION("free derivative function") {
    AD1 x = Variable<0>(2.0);
    AD1 y = Variable<1>(5.0);
    CHECK(derivative(x, 0) == approx(1));
    CHECK(derivative(x, 1) == approx(0));
    CHECK(derivative(y, 1) == approx(1));
    CHECK(derivative(3.0, 0) == approx(0)); // plain number
  }
}

TEST_CASE("Arithmetic: first derivatives", "[autodiff]") {
  AD1 x = Variable<0>(2.0);
  AD1 y = Variable<1>(3.0);

  SECTION("addition") { check(x + y, 5.0, 1, 1); }
  SECTION("subtraction") { check(x - y, -1.0, 1, -1); }
  SECTION("product rule") { check(x * y, 6.0, 3, 2); }
  SECTION("x * x") { check(x * x, 4.0, 4, 0); }

  SECTION("quotient AD / AD") {
    // d/dx = 1/y, d/dy = -x/y^2
    check(x / y, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 9.0);
  }

  SECTION("quotient AD / scalar") { check(x / 4.0, 0.5, 0.25, 0); }

  SECTION("quotient scalar / AD") {
    // d/dy (6/y) = -6/y^2
    check(6.0 / y, 2.0, 0, -6.0 / 9.0);
  }

  SECTION("scalar + AD and scalar - AD") {
    check(1.0 + x, 3.0, 1, 0);
    check(5.0 - y, 2.0, 0, -1);
  }

  SECTION("constants do not contribute to derivatives") {
    AD1 c(10.0);
    check(c * x, 20.0, 10, 0);
    check(c + x, 12.0, 1, 0);
  }

  SECTION("compound expression") {
    // f = x*y + x/y - y ; f_x = y + 1/y ; f_y = x - x/y^2 - 1
    AD1 f = x * y + x / y - y;
    check(f, 6.0 + 2.0 / 3.0 - 3.0, 3.0 + 1.0 / 3.0, 2.0 - 2.0 / 9.0 - 1.0);
  }
}

TEST_CASE("Elementary functions: first derivatives", "[autodiff]") {
  const double t = 0.7;
  AD1 x = Variable<0>(t);

  SECTION("sin") { check(sin(x), std::sin(t), std::cos(t), 0); }
  SECTION("cos") { check(cos(x), std::cos(t), -std::sin(t), 0); }
  SECTION("exp") { check(exp(x), std::exp(t), std::exp(t), 0); }
  SECTION("log") { check(log(x), std::log(t), 1.0 / t, 0); }

  SECTION("sin^2 + cos^2 = 1 with zero derivative") {
    AD1 f = sin(x) * sin(x) + cos(x) * cos(x);
    check(f, 1.0, 0, 0);
  }

  SECTION("exp(log(x)) = x") { check(exp(log(x)), t, 1, 0); }

  SECTION("chain rule: sin(x*y)") {
    AD1 u = Variable<0>(2.0);
    AD1 v = Variable<1>(3.0);
    const double c = std::cos(6.0);
    check(sin(u * v), std::sin(6.0), 3.0 * c, 2.0 * c);
  }

  SECTION("plain doubles still work") {
    CHECK(sin(0.5) == approx(std::sin(0.5)));
    CHECK(cos(0.5) == approx(std::cos(0.5)));
    CHECK(exp(0.5) == approx(std::exp(0.5)));
    CHECK(log(0.5) == approx(std::log(0.5)));
  }
}

TEST_CASE("Gradient agrees with central finite differences", "[autodiff]") {
  const double u = 1.3, v = 0.8, h = 1e-6;

  AD1 x = Variable<0>(u);
  AD1 y = Variable<1>(v);
  AD1 f = func(x, y);

  const double fd_x = (func(u + h, v) - func(u - h, v)) / (2 * h);
  const double fd_y = (func(u, v + h) - func(u, v - h)) / (2 * h);

  CHECK(f.value() == approx(func(u, v)));
  CHECK(f.deriv()[0] == Approx(fd_x).margin(1e-6));
  CHECK(f.deriv()[1] == Approx(fd_y).margin(1e-6));
}

TEST_CASE("Division by zero throws", "[autodiff]") {
  AD1 x = Variable<0>(2.0);
  AD1 zero(0.0);

  CHECK_THROWS_AS(x / zero, std::domain_error);
  CHECK_THROWS_AS(x / 0.0, std::domain_error);
  CHECK_THROWS_AS(1.0 / zero, std::domain_error);
  CHECK_NOTHROW(x / AD1(1.0));
}

TEST_CASE("Output stream operator", "[autodiff]") {
  AD1 x = Variable<0>(1.5);
  std::ostringstream os;
  os << x;
  CHECK(os.str() == "Value: 1.5, Deriv: [1, 0]");
}

TEST_CASE("Nested AutoDiff: second derivatives", "[autodiff][nested]") {
  const double u = 2.0, v = 3.0;
  auto [x, y] = make_nested(u, v);

  // read helpers
  auto val = [](const AD2 &f) { return f.value().value(); };
  auto grad = [](const AD2 &f, size_t i) { return f.deriv()[i].value(); };
  auto hess = [](const AD2 &f, size_t i, size_t j) {
    return f.deriv()[i].deriv()[j];
  };

  SECTION("seeds") {
    CHECK(val(x) == approx(u));
    CHECK(grad(x, 0) == approx(1));
    CHECK(grad(x, 1) == approx(0));
    CHECK(hess(x, 0, 0) == approx(0));
    CHECK(hess(x, 0, 1) == approx(0));
  }

  SECTION("x*y: only mixed second derivative is nonzero") {
    AD2 f = x * y;
    CHECK(val(f) == approx(6.0));
    CHECK(grad(f, 0) == approx(v));
    CHECK(grad(f, 1) == approx(u));
    CHECK(hess(f, 0, 0) == approx(0));
    CHECK(hess(f, 1, 1) == approx(0));
    CHECK(hess(f, 0, 1) == approx(1));
    CHECK(hess(f, 1, 0) == approx(1));
  }

  SECTION("x*x*x: f'' = 6x") {
    AD2 f = x * x * x;
    CHECK(val(f) == approx(8.0));
    CHECK(grad(f, 0) == approx(12.0));
    CHECK(hess(f, 0, 0) == approx(12.0));
  }

  SECTION("sin(x*y): full Hessian") {
    AD2 f = sin(x * y);
    const double s = std::sin(u * v), c = std::cos(u * v);

    CHECK(val(f) == approx(s));
    CHECK(grad(f, 0) == approx(v * c));
    CHECK(grad(f, 1) == approx(u * c));

    CHECK(hess(f, 0, 0) == approx(-s * v * v));
    CHECK(hess(f, 1, 1) == approx(-s * u * u));
    CHECK(hess(f, 0, 1) == approx(c - u * v * s));
    CHECK(hess(f, 1, 0) == approx(c - u * v * s));
  }

  SECTION("inner and outer first derivatives agree") {
    AD2 f = sin(x * y);
    for (size_t i = 0; i < 2; i++)
      CHECK(f.value().deriv()[i] == approx(f.deriv()[i].value()));
  }

  SECTION("Hessian is symmetric") {
    AD2 f = sin(x * y) * exp(x) + log(y);
    CHECK(hess(f, 0, 1) == approx(hess(f, 1, 0)));
  }

  SECTION("exp(x*y)") {
    AD2 f = exp(x * y);
    const double e = std::exp(u * v);
    CHECK(hess(f, 0, 0) == approx(v * v * e));
    CHECK(hess(f, 1, 1) == approx(u * u * e));
    CHECK(hess(f, 0, 1) == approx(e + u * v * e));
  }

  SECTION("log(x*y): f_xx = -1/x^2, f_xy = 0") {
    AD2 f = log(x * y);
    CHECK(grad(f, 0) == approx(1.0 / u));
    CHECK(hess(f, 0, 0) == approx(-1.0 / (u * u)));
    CHECK(hess(f, 1, 1) == approx(-1.0 / (v * v)));
    CHECK(hess(f, 0, 1) == approx(0));
  }

  SECTION("exp(x+x): f'' = 4 exp(2x)") {
    AD2 f = exp(x + x);
    CHECK(hess(f, 0, 0) == approx(4.0 * std::exp(2.0 * u)));
  }
}

TEST_CASE("Nested AutoDiff: Hessian vs finite differences",
          "[autodiff][nested]") {
  const double u = 0.9, v = 1.4, h = 1e-4;
  auto [x, y] = make_nested(u, v);

  AD2 f = sin(x * y) * exp(x) + log(y);
  auto g = [](double a, double b) {
    return std::sin(a * b) * std::exp(a) + std::log(b);
  };

  const double fxx = (g(u + h, v) - 2 * g(u, v) + g(u - h, v)) / (h * h);
  const double fyy = (g(u, v + h) - 2 * g(u, v) + g(u, v - h)) / (h * h);
  const double fxy =
      (g(u + h, v + h) - g(u + h, v - h) - g(u - h, v + h) + g(u - h, v - h)) /
      (4 * h * h);

  CHECK(f.deriv()[0].deriv()[0] == Approx(fxx).margin(1e-5));
  CHECK(f.deriv()[1].deriv()[1] == Approx(fyy).margin(1e-5));
  CHECK(f.deriv()[0].deriv()[1] == Approx(fxy).margin(1e-5));
}
