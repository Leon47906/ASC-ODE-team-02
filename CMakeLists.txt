cmake_minimum_required(VERSION 3.20)
project(ASC_ode)

set (CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(src nanoblas/src)

add_subdirectory (nanoblas)

find_package(Catch2 3 REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable (test_ode demos/test_ode.cpp)
target_link_libraries (test_ode PUBLIC nanoblas)

add_executable (demo_autodiff demos/demo_autodiff.cpp)
target_link_libraries (demo_autodiff PUBLIC nanoblas)

add_executable (test_rc_circuit demos/test_rc_circuit.cpp)
target_link_libraries(test_rc_circuit PUBLIC nanoblas)

add_executable (test_convergence demos/mass_spring_convergence.cpp)
target_link_libraries (test_convergence PUBLIC nanoblas Catch2::Catch2WithMain)

add_executable(test_autodiff demos/test_autodiff.cpp)
target_link_libraries(test_autodiff PUBLIC nanoblas Catch2::Catch2WithMain)
