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

#ifndef IREE_COMPILER_UTILS_TRANSLATIONUTILS_H_
#define IREE_COMPILER_UTILS_TRANSLATIONUTILS_H_

#include <functional>
#include <memory>

#include "iree/compiler/IR/StructureOps.h"
#include "iree/schemas/executable_def_generated.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/llvm/llvm/include/llvm/ADT/StringMap.h"
#include "third_party/llvm/llvm/include/llvm/ADT/StringRef.h"

namespace mlir {
namespace iree_compiler {

// Common translation options for diagnostics and debugging.
struct TranslationOptions {
  // Enables MLIR IR printing during translation.
  // This can be specified via the -print-ir-before-all and -print-ir-after-all
  // command line flags or overridden programmatically via this flag.
  bool print_mlir = false;

  void CopyFrom(const TranslationOptions &other) {
    print_mlir = other.print_mlir;
  }
};

// Options for iree.module translation for diagnostics and debugging.
struct ModuleTranslationOptions : public TranslationOptions {
  // Defines which backend translators will be used to translate executables.
  // If empty then all linked in translators will be used.
  // TODO(benvanik): extend to allow specifying entire config blobs via mlir.
  std::vector<std::string> target_backends;
};

// Options for iree.executable translation for diagnostics and debugging.
// Target configuration is sourced from the iree.target_config op within the
// iree.executable.
struct ExecutableTranslationOptions : public TranslationOptions {};

// Results of a translation operation.
// May contain zero or more executable defs depending on translation options,
// defined target configs, and support.
struct ExecutableTranslationResult {
  std::vector<std::unique_ptr<iree::ExecutableDefT>> executable_defs;
};

// Registered function that given a set of |executableOps| containing one
// or more iree.executables will produce zero or more serialized executables.
//
// Each iree.executable provided contains one iree.executable_target_config with
// backend-specific translation information. The translator can decide whether
// to translate each independently, group them together, etc.
//
// The provided |executableOps| can be mutated by the callee and will be
// preserved for debugging after translation. If any executable in
// |executableOps| is not used by the translator then it should be erased.
using TranslateExecutableFn =
    std::function<llvm::Optional<ExecutableTranslationResult>(
        ArrayRef<IREE::ExecutableOp> executableOps,
        ExecutableTranslationOptions options)>;

// Registers an executable translation function.
struct ExecutableTranslationRegistration {
  ExecutableTranslationRegistration(llvm::StringRef name,
                                    const TranslateExecutableFn &fn);
};

// Returns a read-only reference to the translator registry.
const llvm::StringMap<TranslateExecutableFn>
    &getExecutableTranslationRegistry();

// Returns executable translation backend names matching the given pattern.
// This accepts wildcards for any delimited value. For example, 'foo-*-bar' will
// match 'foo-123-bar' and 'foo-456-bar' and 'foo-10?' will match 'foo-101' and
// 'foo-102'.
std::vector<std::string> matchExecutableTranslationBackendNames(
    llvm::StringRef pattern);

// Creates a new pass manager initialized with the given options.
std::unique_ptr<PassManager> createPassManager(
    MLIRContext *ctx, const TranslationOptions &translationOptions);

// Runs an initialized set of passes on the given module.
LogicalResult runPassPipeline(const TranslationOptions &translationOptions,
                              PassManager *passManager, ModuleOp module);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_TRANSLATIONUTILS_H_
