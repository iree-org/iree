// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/cts/util/registry.h"

#include <algorithm>
#include <iostream>
#include <mutex>

namespace iree::async::cts {
namespace {

// Global storage for registered suites and backends.
// Protected by a mutex for thread-safe static initialization.
struct RegistryData {
  std::mutex mutex;
  std::vector<TestSuiteInfo> test_suites;
  std::vector<BenchmarkSuiteInfo> benchmark_suites;
  std::vector<BackendConfig> backends;
};

RegistryData& GetRegistryData() {
  static RegistryData data;
  return data;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void CtsRegistry::RegisterSuite(TestSuiteInfo info) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);
  data.test_suites.push_back(std::move(info));
}

void CtsRegistry::RegisterBenchmark(BenchmarkSuiteInfo info) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);
  data.benchmark_suites.push_back(std::move(info));
}

void CtsRegistry::RegisterBackend(BackendConfig config) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);
  data.backends.push_back(std::move(config));
}

//===----------------------------------------------------------------------===//
// Instantiation
//===----------------------------------------------------------------------===//

bool CtsRegistry::TagsMatch(const BackendConfig& backend,
                            const std::vector<std::string>& required,
                            const std::vector<std::string>& excluded) {
  // Check required tags: backend must have ALL of them.
  for (const auto& tag : required) {
    if (std::find(backend.tags.begin(), backend.tags.end(), tag) ==
        backend.tags.end()) {
      return false;  // Missing required tag.
    }
  }

  // Check excluded tags: backend must have NONE of them.
  for (const auto& tag : excluded) {
    if (std::find(backend.tags.begin(), backend.tags.end(), tag) !=
        backend.tags.end()) {
      return false;  // Has excluded tag.
    }
  }

  return true;
}

void CtsRegistry::InstantiateAll() {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);

  if (data.backends.empty() ||
      (data.test_suites.empty() && data.benchmark_suites.empty())) {
    return;
  }

  // Accumulate matching backends for each test suite, then finalize.
  for (const auto& suite : data.test_suites) {
    for (const auto& backend : data.backends) {
      if (TagsMatch(backend, suite.required_tags, suite.excluded_tags)) {
        suite.accumulator(backend);
      }
    }
    // Finalize: create gtest instances for all accumulated backends.
    suite.finalizer(suite.file, suite.line);
  }

  // Instantiate benchmark suites for matching backends.
  for (const auto& benchmark : data.benchmark_suites) {
    for (const auto& backend : data.backends) {
      if (TagsMatch(backend, benchmark.required_tags,
                    benchmark.excluded_tags)) {
        benchmark.instantiator(backend);
      }
    }
  }
}

void CtsRegistry::Instantiate(const char* suite_name,
                              const char* backend_name) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);

  // Find the specified suite.
  const TestSuiteInfo* suite = nullptr;
  for (const auto& s : data.test_suites) {
    if (std::string(s.name) == suite_name) {
      suite = &s;
      break;
    }
  }
  if (!suite) {
    std::cerr << "CtsRegistry::Instantiate: Suite '" << suite_name
              << "' not found.\n";
    return;
  }

  // Find the specified backend.
  const BackendConfig* backend = nullptr;
  for (const auto& b : data.backends) {
    if (std::string(b.name) == backend_name) {
      backend = &b;
      break;
    }
  }
  if (!backend) {
    std::cerr << "CtsRegistry::Instantiate: Backend '" << backend_name
              << "' not found.\n";
    return;
  }

  // Check tag compatibility.
  if (!TagsMatch(*backend, suite->required_tags, suite->excluded_tags)) {
    std::cerr << "CtsRegistry::Instantiate: Tags don't match for suite '"
              << suite_name << "' with backend '" << backend_name << "'.\n";
    return;
  }

  suite->accumulator(*backend);
  suite->finalizer(suite->file, suite->line);
}

//===----------------------------------------------------------------------===//
// Introspection
//===----------------------------------------------------------------------===//

std::vector<std::string> CtsRegistry::ListSuites() {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);

  std::vector<std::string> names;
  names.reserve(data.test_suites.size());
  for (const auto& suite : data.test_suites) {
    names.push_back(suite.name);
  }
  return names;
}

std::vector<std::string> CtsRegistry::ListBackends() {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);

  std::vector<std::string> names;
  names.reserve(data.backends.size());
  for (const auto& backend : data.backends) {
    names.push_back(backend.name);
  }
  return names;
}

}  // namespace iree::async::cts
