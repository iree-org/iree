# Conformance Test Suite (CTS) for HAL implementations

These tests exercise IREE's Hardware Abstraction Layer (HAL) in a way that
checks for conformance across implementations and devices. The tests themselves
are structured to help with HAL driver development by using individual features
in isolation, demonstrating typical full-system usage, and pointing out where
capabilities are optional.

## Usage

Each HAL driver (in-tree or out-of-tree) can use the `iree_hal_cts_test_suite()`
CMake function to create a set of tests. See the documentation in
[iree_hal_cts_test_suite.cmake](../../build_tools/cmake/iree_hal_cts_test_suite.cmake)
and [cts_test_base.h](cts_test_base.h) for concrete details.

## On testing for error conditions

In general, error states are only lightly tested because the low level APIs that
IREE's HAL is designed to thinly abstract over often assume programmer usage
will be correct and treat errors as undefined behavior. See the Vulkan spec:

* https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap3.html#introduction-conventions
* https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap4.html#fundamentals-errors

While the generic tests in the CTS may not be able to check for error conditions
exhaustively, individual HAL implementations can implement stricter behavior
or enable higher level checks like what the
[Vulkan Validation Layers](https://github.com/KhronosGroup/Vulkan-ValidationLayers)
provide.

## Tips for adding new HAL implementations

* Driver (`iree_hal_driver_t`) and device (`iree_hal_device_t`) creation, tested
  in [driver_test](driver_test.h), are both prerequisites for all tests.
* Tests for individual components (e.g.
  [descriptor_set_layout_test](descriptor_set_layout_test.h)) are more
  approachable than tests which use collections of components together (e.g.
  [command_buffer_test](command_buffer_test.h)).
