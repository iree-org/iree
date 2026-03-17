// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS Registry for link-time test and benchmark composition.
//
// This registry enables test suites to be compiled once and linked against
// multiple carrier backends, eliminating per-backend instantiation boilerplate.
//
// Architecture:
//   1. Test suites register themselves at static init time via
//      CTS_REGISTER_TEST_SUITE().
//   2. Backends register themselves at static init time via
//      CtsRegistry::RegisterBackend().
//   3. main() calls CtsRegistry::InstantiateAll() before RUN_ALL_TESTS().
//   4. InstantiateAll() creates gtest parameterized test instances for each
//      suite x backend pair that passes tag filtering.
//
// Tag filtering:
//   - required_tags: Backend must have ALL of these tags.
//   - excluded_tags: Backend must have NONE of these tags.
//
// Example test suite registration:
//   CTS_REGISTER_TEST_SUITE(LifecycleTest);  // All backends
//   CTS_REGISTER_TEST_SUITE_WITH_TAGS(ReliabilityTest, {"reliable"}, {});
//
// Example backend registration:
//   CtsRegistry::RegisterBackend({
//       "loopback",
//       CreateLoopbackCarrierPair,
//       {"reliable", "ordered", "zerocopy_tx"}
//   });

#ifndef IREE_NET_CARRIER_CTS_UTIL_REGISTRY_H_
#define IREE_NET_CARRIER_CTS_UTIL_REGISTRY_H_

#include <functional>
#include <string>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/base/status.h"
#include "iree/net/carrier.h"
#include "iree/testing/gtest.h"

// Forward declarations for factory-level CTS support.
// Only pointer types are used in BackendInfo, so full headers are not needed.
typedef struct iree_net_transport_factory_t iree_net_transport_factory_t;
typedef struct iree_net_listener_t iree_net_listener_t;

namespace iree::net::carrier::cts {

//===----------------------------------------------------------------------===//
// Carrier pair factory
//===----------------------------------------------------------------------===//

// A connected pair of carriers returned by the factory.
// The factory is responsible for creating proactor, establishing connection,
// and returning both endpoints ready for activation.
struct CarrierPair {
  iree_net_carrier_t* client;
  iree_net_carrier_t* server;
  iree_async_proactor_t* proactor;  // Shared by both carriers.

  // Optional cleanup context for pair-specific resources (e.g., RDMA QP state).
  void* context;
  void (*cleanup)(void* context);
};

// Factory function that creates a connected carrier pair.
//
// |proactor| is optional. When non-NULL the factory creates carriers on the
// caller's proactor (the production pattern for connection pools, reconnection,
// server accept loops). When NULL the factory creates its own proactor and
// returns it via pair.proactor.
//
// Returns UNAVAILABLE if the backend cannot be created on this system
// (e.g., no RDMA devices). Tests skip gracefully on UNAVAILABLE.
using CarrierPairFactory =
    std::function<iree::StatusOr<CarrierPair>(iree_async_proactor_t*)>;

// Identifies a carrier backend for test parameterization.
struct BackendInfo {
  const char* name;            // Human-readable: "tcp", "loopback", etc.
  CarrierPairFactory factory;  // Creates a connected carrier pair.

  // Factory-level CTS support. When provided, factory test suites exercise
  // the transport factory → listener → connection → carrier pipeline.
  // Null functions indicate no factory support (test suites skip via tag
  // filtering — backends without these populate tags accordingly).

  // Creates the transport factory under test.
  std::function<iree_status_t(iree_allocator_t, iree_net_transport_factory_t**)>
      create_factory;

  // Returns a unique listener bind address. Each call produces a fresh address
  // so tests can create multiple independent listeners.
  std::function<std::string()> make_bind_address;

  // Returns the connect address for a listener created with |bind_address|.
  // For transports that assign addresses dynamically (e.g., TCP ephemeral
  // ports), this queries the listener for its actual address.
  std::function<std::string(const std::string&, iree_net_listener_t*)>
      resolve_connect_address;

  // Returns an address where no listener exists (for error-path tests).
  std::function<std::string(iree_async_proactor_t*)> make_unreachable_address;
};

// Returns human-readable test suffix from BackendInfo.
// Used as the generator for INSTANTIATE_TEST_SUITE_P.
struct BackendName {
  std::string operator()(
      const ::testing::TestParamInfo<BackendInfo>& info) const {
    return info.param.name;
  }
};

// Printer for BackendInfo in gtest assertions/failure messages.
// Without this, gtest prints the raw bytes of the struct.
inline void PrintTo(const BackendInfo& info, std::ostream* os) {
  *os << info.name;
}

//===----------------------------------------------------------------------===//
// Backend configuration with tags
//===----------------------------------------------------------------------===//

// Identifies a backend configuration for test instantiation.
// Extends BackendInfo with tags for filtering which tests apply.
struct BackendConfig {
  const char* name;               // "tcp", "loopback", "rdma_rc", etc.
  BackendInfo info;               // Factory function.
  std::vector<std::string> tags;  // {"reliable", "ordered", "zerocopy_tx", ...}
};

//===----------------------------------------------------------------------===//
// Test suite registration
//===----------------------------------------------------------------------===//

// Function called for each matching backend to accumulate backends for a suite.
using TestSuiteAccumulator = std::function<void(const BackendConfig& config)>;

// Function called once after all backends are accumulated to create gtest
// instances.
using TestSuiteFinalizer = std::function<void(const char* file, int line)>;

// Metadata about a registered test suite.
struct TestSuiteInfo {
  const char* name;
  TestSuiteAccumulator accumulator;
  TestSuiteFinalizer finalizer;
  const char* file;
  int line;

  // Required tags: suite only instantiated for backends with ALL of these.
  std::vector<std::string> required_tags;

  // Excluded tags: suite skipped for backends with ANY of these.
  std::vector<std::string> excluded_tags;
};

//===----------------------------------------------------------------------===//
// Benchmark registration
//===----------------------------------------------------------------------===//

// Function that registers benchmarks for a given backend config.
using BenchmarkInstantiator = std::function<void(const BackendConfig& config)>;

// Metadata about a registered benchmark suite.
struct BenchmarkSuiteInfo {
  const char* name;
  BenchmarkInstantiator instantiator;
  std::vector<std::string> required_tags;
  std::vector<std::string> excluded_tags;
};

//===----------------------------------------------------------------------===//
// CTS Registry
//===----------------------------------------------------------------------===//

class CtsRegistry {
 public:
  //===--------------------------------------------------------------------===//
  // Registration (called at static init time)
  //===--------------------------------------------------------------------===//

  // Register a test suite. Called by CTS_REGISTER_TEST_SUITE macro.
  static void RegisterSuite(TestSuiteInfo info);

  // Register a benchmark suite. Called by CTS_REGISTER_BENCHMARK macro.
  static void RegisterBenchmark(BenchmarkSuiteInfo info);

  // Register a backend configuration. Called by backend factory files.
  static void RegisterBackend(BackendConfig config);

  //===--------------------------------------------------------------------===//
  // Instantiation (called from main, before RUN_ALL_TESTS)
  //===--------------------------------------------------------------------===//

  // Instantiate all suites x all backends, respecting tag filters.
  static void InstantiateAll();

  // Instantiate specific suite for specific backend (for debugging).
  static void Instantiate(const char* suite_name, const char* backend_name);

  //===--------------------------------------------------------------------===//
  // Backend lookup
  //===--------------------------------------------------------------------===//

  // Returns the first registered backend, or nullptr if none registered.
  // Fuzz binaries link exactly one backend and use this to retrieve it.
  static const BackendConfig* GetBackend();

  //===--------------------------------------------------------------------===//
  // Introspection (for tooling, test listing)
  //===--------------------------------------------------------------------===//

  static std::vector<std::string> ListSuites();
  static std::vector<std::string> ListBackends();

 private:
  // Check if backend has all required tags and none of the excluded tags.
  static bool TagsMatch(const BackendConfig& backend,
                        const std::vector<std::string>& required,
                        const std::vector<std::string>& excluded);
};

//===----------------------------------------------------------------------===//
// Test Registration Helpers
//===----------------------------------------------------------------------===//

namespace internal {

// Name generator function for test instantiation.
inline std::string GetBackendName(
    const ::testing::TestParamInfo<BackendInfo>& info) {
  return BackendName()(info);
}

}  // namespace internal

//===----------------------------------------------------------------------===//
// Shared carrier lifecycle utilities
//===----------------------------------------------------------------------===//

// Null recv handler that discards all received data.
// Releases the buffer lease to return it to the pool for reuse.
inline iree_status_t NullRecvHandler(void* user_data, iree_async_span_t data,
                                     iree_async_buffer_lease_t* lease) {
  iree_async_buffer_lease_release(lease);
  return iree_ok_status();
}

inline iree_net_carrier_recv_handler_t MakeNullRecvHandler() {
  return {NullRecvHandler, nullptr};
}

// Deactivates a carrier and polls the proactor until draining completes.
// Safe to call on carriers in any state:
//   CREATED/DEACTIVATED: returns immediately.
//   ACTIVE: initiates deactivation, then polls until complete.
//   DRAINING: deactivation already in progress; polls until complete.
//
// IMPORTANT: Must be called from the proactor's poll owner thread since this
// function calls iree_async_proactor_poll() internally.
inline void DeactivateAndDrain(iree_net_carrier_t* carrier,
                               iree_async_proactor_t* proactor) {
  iree_net_carrier_state_t state = iree_net_carrier_state(carrier);
  if (state == IREE_NET_CARRIER_STATE_CREATED ||
      state == IREE_NET_CARRIER_STATE_DEACTIVATED) {
    return;
  }

  // Only initiate deactivation from ACTIVE state. If already DRAINING (a
  // previous deactivate call is in progress), skip straight to polling.
  if (state == IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_status_t status =
        iree_net_carrier_deactivate(carrier, nullptr, nullptr);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return;
    }
  }

  // Poll until the carrier reaches DEACTIVATED (all pending operations
  // drained).
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(1000);
  while (iree_net_carrier_state(carrier) !=
         IREE_NET_CARRIER_STATE_DEACTIVATED) {
    if (iree_time_now() >= deadline) break;
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor, iree_make_timeout_ms(10), &completed);
    iree_status_ignore(status);
  }
}

//===----------------------------------------------------------------------===//
// Test Registration Macros
//===----------------------------------------------------------------------===//

// Basic registration - suite runs against all backends.
#define CTS_REGISTER_TEST_SUITE(TestClass) \
  CTS_REGISTER_TEST_SUITE_WITH_TAGS(TestClass, {}, {})

// Registration with tag requirements.
// required_tags: backend must have ALL of these tags
// excluded_tags: backend must have NONE of these tags
//
// Each test suite has a static vector of BackendInfos that accumulates
// matching backends during InstantiateAll(). After all backends are added,
// the suite is instantiated once with ValuesIn() covering all backends.
#define CTS_REGISTER_TEST_SUITE_WITH_TAGS(TestClass, required_tags,         \
                                          excluded_tags)                    \
  namespace {                                                               \
  struct TestClass##_Backends {                                             \
    static std::vector<::iree::net::carrier::cts::BackendInfo>& Get() {     \
      static std::vector<::iree::net::carrier::cts::BackendInfo> backends;  \
      return backends;                                                      \
    }                                                                       \
    static ::testing::internal::ParamGenerator<                             \
        ::iree::net::carrier::cts::BackendInfo>                             \
    Generator() {                                                           \
      return ::testing::ValuesIn(Get());                                    \
    }                                                                       \
  };                                                                        \
  }                                                                         \
  static bool TestClass##_registered_ =                                     \
      (::iree::net::carrier::cts::CtsRegistry::RegisterSuite(               \
           {#TestClass,                                                     \
            [](const ::iree::net::carrier::cts::BackendConfig& cfg) {       \
              TestClass##_Backends::Get().push_back(cfg.info);              \
            },                                                              \
            [](const char* file, int line) {                                \
              if (TestClass##_Backends::Get().empty()) return;              \
              ::testing::UnitTest::GetInstance()                            \
                  ->parameterized_test_registry()                           \
                  .GetTestSuitePatternHolder<TestClass>(                    \
                      #TestClass,                                           \
                      ::testing::internal::CodeLocation(file, line))        \
                  ->AddTestSuiteInstantiation(                              \
                      "CTS", &TestClass##_Backends::Generator,              \
                      &::iree::net::carrier::cts::internal::GetBackendName, \
                      file, line);                                          \
            },                                                              \
            __FILE__, __LINE__, required_tags, excluded_tags}),             \
       true)

//===----------------------------------------------------------------------===//
// Benchmark Suite Registration Macros
//===----------------------------------------------------------------------===//

// Basic benchmark suite registration - runs against all backends.
// The BenchmarkClass must have a static method:
//   static void RegisterBenchmarks(const char* prefix,
//                                   const CarrierPairFactory& factory);
#define CTS_REGISTER_BENCHMARK_SUITE(BenchmarkClass) \
  CTS_REGISTER_BENCHMARK_SUITE_WITH_TAGS(BenchmarkClass, {}, {})

// Benchmark suite registration with tag requirements.
// required_tags: backend must have ALL of these tags.
// excluded_tags: backend must have NONE of these tags.
#define CTS_REGISTER_BENCHMARK_SUITE_WITH_TAGS(BenchmarkClass, required_tags, \
                                               excluded_tags)                 \
  static bool BenchmarkClass##_benchmark_registered_ =                        \
      (::iree::net::carrier::cts::CtsRegistry::RegisterBenchmark(             \
           {#BenchmarkClass,                                                  \
            [](const ::iree::net::carrier::cts::BackendConfig& cfg) {         \
              BenchmarkClass::RegisterBenchmarks(cfg.name, cfg.info.factory); \
            },                                                                \
            required_tags, excluded_tags}),                                   \
       true)

}  // namespace iree::net::carrier::cts

#endif  // IREE_NET_CARRIER_CTS_UTIL_REGISTRY_H_
