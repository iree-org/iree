// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/util/registry.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace iree::hal::cts {
namespace {

// Global storage for registered suites and backends.
// Protected by a mutex for thread-safe static initialization.
// Executable format registered before its backend (or separately from it).
struct PendingFormat {
  std::string backend_name;
  ExecutableFormat format;
};

struct RegistryData {
  std::mutex mutex;
  bool instantiated = false;
  std::vector<TestSuiteInfo> test_suites;
  std::vector<BackendConfig> backends;
  std::vector<PendingFormat> pending_formats;
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

void CtsRegistry::RegisterBackend(BackendConfig config) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);
  data.backends.push_back(std::move(config));
}

void CtsRegistry::RegisterExecutableFormat(const char* backend_name,
                                           ExecutableFormat format) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);
  data.pending_formats.push_back(
      {std::string(backend_name), std::move(format)});
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

// Merges pending executable formats into their matching backends.
// Must be called with data.mutex held.
static void MergePendingFormats(RegistryData& data) {
  for (auto& pending : data.pending_formats) {
    bool found = false;
    for (auto& backend : data.backends) {
      if (pending.backend_name == backend.name) {
        backend.executable_formats.push_back(std::move(pending.format));
        found = true;
        break;
      }
    }
    if (!found) {
      std::cerr << "CtsRegistry: Executable format for '"
                << pending.backend_name
                << "' has no matching backend registration.\n";
      std::abort();
    }
  }
  data.pending_formats.clear();
}

void CtsRegistry::InstantiateAll() {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);

  if (data.instantiated) {
    std::cerr << "CtsRegistry::InstantiateAll called twice.\n";
    std::abort();
  }
  data.instantiated = true;

  // Merge pending executable formats into their backends. This handles the
  // case where RegisterExecutableFormat() was called before or after the
  // corresponding RegisterBackend() — static init ordering is unspecified.
  MergePendingFormats(data);

  if (data.backends.empty() || data.test_suites.empty()) {
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
}

void CtsRegistry::Instantiate(const char* suite_name,
                              const char* backend_name) {
  auto& data = GetRegistryData();
  std::lock_guard<std::mutex> lock(data.mutex);

  // Merge pending executable formats (same as InstantiateAll) so that
  // formats registered separately from backends are available.
  MergePendingFormats(data);

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
    std::abort();
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
    std::abort();
  }

  // Check tag compatibility.
  if (!TagsMatch(*backend, suite->required_tags, suite->excluded_tags)) {
    std::cerr << "CtsRegistry::Instantiate: Tags don't match for suite '"
              << suite_name << "' with backend '" << backend_name << "'.\n";
    std::abort();
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

}  // namespace iree::hal::cts
