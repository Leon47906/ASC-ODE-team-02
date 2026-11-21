#ifndef TIMERSTEPPER_HPP
#define TIMERSTEPPER_HPP

#include <exception>
#include <functional>
#include <memory>

#include "Newton.hpp"
#include "vector.hpp"

namespace ASC_ode {

// base class of all time-stepping methods for y' = rhs(y)
class TimeStepper {
protected:
  std::shared_ptr<NonlinearFunction> m_rhs;

public:
  TimeStepper(std::shared_ptr<NonlinearFunction> rhs) : m_rhs(rhs) {}
  virtual ~TimeStepper() = default;
  // advances y by one time-step tau
  virtual void doStep(double tau, VectorView<double> y) = 0;
};

// y_{n+1} = y_n + tau f(y_n)
class ExplicitEuler : public TimeStepper {
  Vector<> m_vecf; // allocated once, not in every step
public:
  ExplicitEuler(std::shared_ptr<NonlinearFunction> rhs)
      : TimeStepper(rhs), m_vecf(rhs->dimF()) {}
  void doStep(double tau, VectorView<double> y) override {
    this->m_rhs->evaluate(y, m_vecf);
    y += tau * m_vecf;
  }
};

// solves y - y_n - tau f(y) = 0 with Newton's method. The equation is built
// once by the function algebra, y_n and tau are updated in every step.
class ImplicitEuler : public TimeStepper {
  std::shared_ptr<NonlinearFunction> m_equ;
  std::shared_ptr<Parameter> m_tau;
  std::shared_ptr<ConstantFunction> m_yold;

public:
  ImplicitEuler(std::shared_ptr<NonlinearFunction> rhs)
      : TimeStepper(rhs), m_tau(std::make_shared<Parameter>(0.0)) {
    m_yold = std::make_shared<ConstantFunction>(rhs->dimX());
    auto ynew = std::make_shared<IdentityFunction>(rhs->dimX());
    m_equ = ynew - m_yold - m_tau * m_rhs;
  }

  void doStep(double tau, VectorView<double> y) override {
    m_yold->set(y);
    m_tau->set(tau);
    NewtonSolver(m_equ, y);
  }
};

class ImprovedEuler : public TimeStepper {
  Vector<> m_vecf;
  Vector<> ytilde;

public:
  ImprovedEuler(std::shared_ptr<NonlinearFunction> rhs)
      : TimeStepper(rhs), m_vecf(rhs->dimF()), ytilde(rhs->dimX()) {};
  void doStep(double tau, VectorView<double> y) override {
    this->m_rhs->evaluate(y, m_vecf);
    ytilde = y + tau / 2.0 * m_vecf;
    this->m_rhs->evaluate(ytilde, m_vecf);
    y += tau * m_vecf;
  }
};

class CrankNicolson : public TimeStepper {
  Vector<> m_vecf;
  std::shared_ptr<NonlinearFunction> m_equ;
  std::shared_ptr<Parameter> m_tauhalf;
  std::shared_ptr<ConstantFunction> m_yold, m_fyold;

public:
  CrankNicolson(std::shared_ptr<NonlinearFunction> rhs)
      : TimeStepper(rhs), m_vecf(rhs->dimF()),
        m_tauhalf(std::make_shared<Parameter>(0.0)) {
    m_fyold = std::make_shared<ConstantFunction>(rhs->dimF());
    m_yold = std::make_shared<ConstantFunction>(rhs->dimX());
    auto ynew = std::make_shared<IdentityFunction>(rhs->dimX());
    m_equ = ynew - m_yold - m_tauhalf * m_rhs - m_tauhalf * m_fyold;
  }
  void doStep(double tau, VectorView<double> y) override {
    m_yold->set(y);
    m_tauhalf->set(tau / 2.0);
    this->m_rhs->evaluate(y, m_vecf);
    m_fyold->set(m_vecf);
    NewtonSolver(m_equ, y);
  }
};

} // namespace ASC_ode

#endif
