#include <autodiff.hpp>
#include <iostream>

using namespace ASC_ode;

template <typename T> T func1(T x, T y) { return x * sin(y); }

int main() {
  double x = 1, y = 2;
  AutoDiff<2> adx = Variable<0>(x);
  AutoDiff<2> ady = Variable<1>(y);

  std::cout << "adx = " << adx << std::endl;
  std::cout << "ady = " << ady << std::endl;

  AutoDiff<2> prod = adx * ady;
  std::cout << "prod = " << prod << std::endl;

  std::cout << "func1(adx, ady) = " << func1(adx, ady) << std::endl;

  double eps = 1e-8;
  std::cout << "numdiff df/dx = "
            << (func1(x + eps, y) - func1(x - eps, y)) / (2 * eps) << std::endl;
  std::cout << "numdiff df/dy = "
            << (func1(x, y + eps) - func1(x, y - eps)) / (2 * eps) << std::endl;

  {
    // we can do second derivatives:
    AutoDiff<1, AutoDiff<1>> addx{Variable<0>(2)};
    std::cout << "addx = " << addx << std::endl;
    // func = x*x
    // func' = 2*x
    // func'' = 2
    std::cout << "addx*addx = " << addx * addx << std::endl;

    double u = 1, v = 2;
    AutoDiff<2> adu = Variable<0>(u);
    AutoDiff<2> adv = Variable<1>(v);

    AutoDiff<2, AutoDiff<2>> addu = Variable<0, AutoDiff<2>>(adu);
    AutoDiff<2, AutoDiff<2>> addv = Variable<1, AutoDiff<2>>(adv);

    AutoDiff<2, AutoDiff<2>> f = log(sin(addu * addv));

    std::cout << "f = " << f.value().value() << std::endl;

    for (size_t i = 0; i < 2; i++) {
      std::cout << "f_" << i << " = " << f.deriv()[i].value() << std::endl;
    }

    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 2; j++) {
        std::cout << "f_" << i << j << " = " << f.deriv()[i].deriv()[j]
                  << std::endl;
      }
    }
  }
  return 0;
}
