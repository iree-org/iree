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

#include "iree/compiler/Dialect/HAL/Target/ExecutableTarget.h"

#include <algorithm>

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static llvm::cl::OptionCategory halTargetOptionsCategory(
    "IREE HAL executable target options");

static llvm::cl::list<std::string> targetBackendsFlag{
    "iree-hal-target-backends",
    llvm::cl::desc("Target backends for executable compilation"),
    llvm::cl::ZeroOrMore,
    llvm::cl::cat(halTargetOptionsCategory),
};

ExecutableTargetOptions getExecutableTargetOptionsFromFlags() {
  ExecutableTargetOptions targetOptions;
  targetOptions.targets = targetBackendsFlag;
  return targetOptions;
}

// Returns the static registry of translator names to translation functions.
static llvm::StringMap<ExecutableTargetFn>
    &getMutableExecutableTargetRegistry() {
  static llvm::StringMap<ExecutableTargetFn> registry;
  return registry;
}

ExecutableTargetRegistration::ExecutableTargetRegistration(
    llvm::StringRef name, const ExecutableTargetFn &fn) {
  auto &registry = getMutableExecutableTargetRegistry();
  if (registry.count(name) > 0) {
    llvm::report_fatal_error(
        "Attempting to overwrite an existing translation function");
  }
  assert(fn && "Attempting to register an empty translation function");
  registry[name] = fn;
}

const llvm::StringMap<ExecutableTargetFn> &getExecutableTargetRegistry() {
  return getMutableExecutableTargetRegistry();
}

// Returns true if the given |value| matches |pattern| (normal * and ? rules).
static bool matchPattern(StringRef value, StringRef pattern) {
  size_t nextCharIndex = pattern.find_first_of("*?");
  if (nextCharIndex == std::string::npos) {
    return value == pattern;
  } else if (nextCharIndex > 0) {
    if (value.substr(0, nextCharIndex) != pattern.substr(0, nextCharIndex)) {
      return false;
    }
    value = value.substr(nextCharIndex);
    pattern = pattern.substr(nextCharIndex);
  }
  char patternChar = pattern[0];
  if (value.empty() && pattern.empty()) {
    return true;
  } else if (patternChar == '*' && pattern.size() > 1 && value.empty()) {
    return false;
  } else if (patternChar == '*' && pattern.size() == 1) {
    return true;
  } else if (patternChar == '?' || value[0] == patternChar) {
    return matchPattern(value.substr(1), pattern.substr(1));
  } else if (patternChar == '*') {
    return matchPattern(value, pattern.substr(1)) ||
           matchPattern(value.substr(1), pattern);
  }
  return false;
}

std::vector<std::string> matchExecutableTargetNames(llvm::StringRef pattern) {
  std::vector<std::string> matches;
  for (auto &entry : getExecutableTargetRegistry()) {
    if (matchPattern(entry.getKey(), pattern)) {
      matches.push_back(entry.getKey().str());
    }
  }
  return matches;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
