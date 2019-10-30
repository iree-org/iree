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

#include "iree/compiler/Utils/TranslationUtils.h"

#include "third_party/llvm/llvm/include/llvm/Support/Debug.h"
#include "third_party/llvm/llvm/include/llvm/Support/ErrorHandling.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Returns the static registry of translator names to translation functions.
llvm::StringMap<TranslateExecutableFn>
    &getMutableExecutableTranslationRegistry() {
  static llvm::StringMap<TranslateExecutableFn> registry;
  return registry;
}

// Returns true if the given |value| matches |pattern| (normal * and ? rules).
bool matchPattern(StringRef value, StringRef pattern) {
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

// Force enables IR printing on the |passManager|.
void enableIRPrinting(PassManager *passManager) {
  auto notVerifier = [](Pass *pass) {
    return pass->getName() != "FunctionVerifier" &&
           pass->getName() != "ModuleVerifier";
  };
  bool printModuleScope = false;
  passManager->enableIRPrinting(/*shouldPrintBeforePass=*/{},
                                /*shouldPrintAfterPass=*/notVerifier,
                                printModuleScope, llvm::dbgs());
  passManager->disableMultithreading();
}

}  // namespace

ExecutableTranslationRegistration::ExecutableTranslationRegistration(
    llvm::StringRef name, const TranslateExecutableFn &fn) {
  auto &registry = getMutableExecutableTranslationRegistry();
  if (registry.find(name) != registry.end()) {
    llvm::report_fatal_error(
        "Attempting to overwrite an existing translation function");
  }
  assert(fn && "Attempting to register an empty translation function");
  registry[name] = fn;
}

const llvm::StringMap<TranslateExecutableFn>
    &getExecutableTranslationRegistry() {
  return getMutableExecutableTranslationRegistry();
}

std::vector<std::string> matchExecutableTranslationBackendNames(
    llvm::StringRef pattern) {
  std::vector<std::string> matches;
  for (auto &entry : getExecutableTranslationRegistry()) {
    if (matchPattern(entry.getKey(), pattern)) {
      matches.push_back(entry.getKey().str());
    }
  }
  return matches;
}

std::unique_ptr<PassManager> createPassManager(
    MLIRContext *ctx, const TranslationOptions &translationOptions) {
  std::unique_ptr<PassManager> passManager(new PassManager(ctx));

  // Enable IR printing/timing/etc from command line options.
  registerPassManagerCLOptions();
  applyPassManagerCLOptions(*passManager);

  // Override with programmatic options.
  if (translationOptions.print_mlir) {
    enableIRPrinting(passManager.get());
  }

  return passManager;
}

LogicalResult runPassPipeline(const TranslationOptions &translationOptions,
                              PassManager *passManager, ModuleOp module) {
  if (translationOptions.print_mlir) {
    module.dump();
  }

  // Run on the module.
  if (failed(passManager->run(module))) {
    return failure();
  }

  if (translationOptions.print_mlir) {
    module.dump();
  }

  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
