#include <autodiff.hpp>
#include <fstream>
#include <nonlinfunc.hpp>
#include <ostream>
#include <vector>

using namespace ASC_ode;
// Plot
int main() {
  int points = 101;
  double meshwidth = 2.0 / (points - 1);
  std::vector<double> xs(points, -1);
  for (size_t i = 1; i < points; i++) {
    xs[i] = xs[i - 1] + meshwidth;
  }

  int max_order = 5;
  std::vector<AutoDiff<1>> p(max_order + 1);

  std::ofstream outfile("output_demo_legendre.txt");
  outfile << "x";
  for (int order = 0; order < max_order + 1; order++) {
    outfile << " " << order << " " << order << "'";
  }
  outfile << std::endl;
  for (size_t step = 0; step < points; step++) {
    AutoDiff<1> x = Variable<0>(xs[step]);
    LegendrePolynomials(max_order + 1, x, p);
    outfile << xs[step];
    for (int order = 0; order < max_order + 1; order++) {
      outfile << " " << p[order].value() << " " << p[order].deriv()[0];
    }
    outfile << std::endl;
  }
  return 0;
}
