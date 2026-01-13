// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared test framework for VM module testing.
//
// This framework provides a common test runner that works across different
// VM module implementations (bytecode interpreter, EmitC, JIT, etc.).
// Tests are defined in MLIR files under iree/vm/test/ and compiled to
// different formats per backend.

#ifndef IREE_VM_TESTING_TEST_RUNNER_H_
#define IREE_VM_TESTING_TEST_RUNNER_H_

#include <functional>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"

namespace iree::vm::testing {

//===----------------------------------------------------------------------===//
// VMTestParams
//===----------------------------------------------------------------------===//
// Parameters for a single VM test function.

// Module creation function type.
// Different backends implement this differently:
// - Bytecode: loads from embedded binary data
// - EmitC: calls static <module>_create() function
// - JIT: compiles and loads at runtime
using VMModuleCreateFn =
    std::function<iree_status_t(iree_vm_instance_t*, iree_vm_module_t**)>;

// Parameters describing a single test to run.
struct VMTestParams {
  // Module name (e.g., "arithmetic_ops").
  std::string module_name;
  // Function name within the module (e.g., "test_add_i32").
  std::string function_name;
  // Factory function to create the module under test.
  VMModuleCreateFn create_module;
  // Whether this function is expected to fail (fail_ prefix).
  bool expects_failure = false;
  // Factory functions for prerequisite modules that must be loaded before the
  // module under test. These are loaded in order and added to the context
  // first. Examples: native yieldable test module, HAL module, custom import
  // modules.
  std::vector<VMModuleCreateFn> prerequisite_modules;
};

// Allows test names to be printed nicely in gtest output.
std::ostream& operator<<(std::ostream& os, const VMTestParams& params);

//===----------------------------------------------------------------------===//
// VMTestResources
//===----------------------------------------------------------------------===//
// Static resources shared across all tests in a suite.

class VMTestResources {
 public:
  static iree_vm_instance_t* instance_;
};

//===----------------------------------------------------------------------===//
// VMTestRunner
//===----------------------------------------------------------------------===//
// Base test fixture for VM module testing.
//
// Usage:
//   1. Backend-specific test files include this header
//   2. Backend implements GetTestParams() returning vector<VMTestParams>
//   3. INSTANTIATE_TEST_SUITE_P with the params
//
// The runner automatically:
//   - Creates VM instance/context
//   - Loads the module under test
//   - Optionally loads the native yieldable test module
//   - Executes functions and checks results
//   - Handles async/yieldable functions transparently

template <typename BaseType = ::testing::Test>
class VMTestRunner : public BaseType,
                     public ::testing::WithParamInterface<VMTestParams>,
                     public VMTestResources {
 public:
  static void SetUpTestSuite() {
    IREE_ASSERT_OK(iree_vm_instance_create(
        IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance_));
  }

  static void TearDownTestSuite() {
    if (instance_) {
      iree_vm_instance_release(instance_);
      instance_ = nullptr;
    }
  }

  void SetUp() override {
    const auto& params = this->GetParam();

    // Build module list for context.
    std::vector<iree_vm_module_t*> modules;

    // Create and add prerequisite modules first (in order).
    for (const auto& create_fn : params.prerequisite_modules) {
      iree_vm_module_t* prereq_module = nullptr;
      IREE_ASSERT_OK(create_fn(instance_, &prereq_module));
      prerequisite_modules_.push_back(prereq_module);
      modules.push_back(prereq_module);
    }

    // Create the module under test and add last.
    IREE_ASSERT_OK(params.create_module(instance_, &test_module_));
    modules.push_back(test_module_);

    IREE_ASSERT_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));
  }

  void TearDown() override {
    if (context_) {
      iree_vm_context_release(context_);
      context_ = nullptr;
    }
    if (test_module_) {
      iree_vm_module_release(test_module_);
      test_module_ = nullptr;
    }
    for (auto* module : prerequisite_modules_) {
      iree_vm_module_release(module);
    }
    prerequisite_modules_.clear();
  }

  // Runs a function by name.
  // Handles DEFERRED status by resuming until completion.
  // NOTE: Only supports void-returning functions; test functions should perform
  // internal assertions via vm.check.* ops rather than returning values.
  iree_status_t RunFunction(const char* function_name) {
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
        test_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_make_cstring_view(function_name), &function));

    IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                    iree_vm_context_state_resolver(context_),
                                    iree_allocator_system());

    iree_vm_function_call_t call;
    memset(&call, 0, sizeof(call));
    call.function = function;

    iree_status_t status =
        function.module->begin_call(function.module->self, stack, call);

    // Resume until completion.
    // Limit iterations to catch infinite yield loops in tests.
    constexpr int kMaxResumeCount = 10000;
    int resume_count = 0;
    while (iree_status_code(status) == IREE_STATUS_DEFERRED) {
      iree_status_ignore(status);
      if (++resume_count > kMaxResumeCount) {
        iree_vm_stack_deinitialize(stack);
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "resume limit (%d) exceeded for function '%s'; possible infinite "
            "yield loop",
            kMaxResumeCount, function_name);
      }
      status = function.module->resume_call(function.module->self, stack,
                                            call.results);
    }

    iree_vm_stack_deinitialize(stack);
    return status;
  }

 protected:
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* test_module_ = nullptr;
  std::vector<iree_vm_module_t*> prerequisite_modules_;
};

// Storage for static members.
// Note: This must only be included in one translation unit per test binary.
// The generated test template will include this.
#define IREE_VM_TEST_RUNNER_STATIC_STORAGE()                           \
  namespace iree::vm::testing {                                        \
  /*static*/ iree_vm_instance_t* VMTestResources::instance_ = nullptr; \
  }

//===----------------------------------------------------------------------===//
// Standard Test Macros
//===----------------------------------------------------------------------===//
// The parameterized test that runs each function.

#define IREE_VM_TEST_F(test_class)                                          \
  TEST_P(test_class, Check) {                                               \
    const auto& params = GetParam();                                        \
    iree_status_t status = RunFunction(params.function_name.c_str());       \
    if (iree_status_is_ok(status)) {                                        \
      if (params.expects_failure) {                                         \
        GTEST_FAIL() << "Function expected failure but succeeded";          \
      }                                                                     \
    } else {                                                                \
      if (params.expects_failure) {                                         \
        iree_status_ignore(status);                                         \
      } else {                                                              \
        GTEST_FAIL() << "Function expected success but failed with error: " \
                     << iree::Status(std::move(status)).ToString();         \
      }                                                                     \
    }                                                                       \
  }

}  // namespace iree::vm::testing

#endif  // IREE_VM_TESTING_TEST_RUNNER_H_
