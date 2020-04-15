// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"

#include <algorithm>

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Returns the static registry of translator names to translation functions.
static llvm::StringMap<CreateTargetBackendFn> &getMutableTargetRegistry() {
  static llvm::StringMap<CreateTargetBackendFn> registry;
  return registry;
}

TargetBackendRegistration::TargetBackendRegistration(llvm::StringRef name,
                                                     CreateTargetBackendFn fn) {
  auto &registry = getMutableTargetRegistry();
  if (registry.count(name) > 0) {
    llvm::report_fatal_error(
        "Attempting to overwrite an existing translation backend");
  }
  assert(fn && "Attempting to register an empty translation function");
  registry[name] = std::move(fn);
}

const llvm::StringMap<CreateTargetBackendFn> &getTargetRegistry() {
  return getMutableTargetRegistry();
}

std::vector<std::string> getRegisteredTargetBackends() {
  std::vector<std::string> result;
  for (auto &entry : getTargetRegistry()) {
    result.push_back(entry.getKey().str());
  }
  std::sort(result.begin(), result.end(),
            [](const auto &a, const auto &b) { return a < b; });
  return result;
}

std::vector<std::unique_ptr<TargetBackend>> matchTargetBackends(
    ArrayRef<std::string> patterns) {
  std::vector<std::unique_ptr<TargetBackend>> matches;
  for (auto pattern : patterns) {
    for (auto &entry : getTargetRegistry()) {
      if (TargetBackend::matchPattern(entry.getKey(), pattern)) {
        matches.push_back(entry.getValue()());
      }
    }
  }

  // To ensure deterministic builds we sort matches by name.
  std::sort(matches.begin(), matches.end(),
            [](const auto &a, const auto &b) { return a->name() < b->name(); });
  return matches;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
