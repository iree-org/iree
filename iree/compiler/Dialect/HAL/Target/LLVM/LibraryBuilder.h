// Copyright 2021 Google LLC
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
//

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LIBRARYBUILDER_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LIBRARYBUILDER_H_

#include <string>

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTargetOptions.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Module.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Defines an `iree_hal_executable_library_v0_t` and builds the runtime metadata
// structures and query functions.
//
// See iree/hal/local/executable_library.h for more information.
//
// Usage:
//  LibraryBuilder builder(&module);
//  builder.addEntryPoint("hello", "", &helloFunc);
//  ...
//  auto *queryFunc = builder.build("_query_library_foo");
//  // call queryFunc, export it, etc
class LibraryBuilder {
 public:
  // Builder mode setting.
  enum class Mode : uint32_t {
    // Include entry point names and tags.
    // If not specified then the reflection strings will be excluded to reduce
    // binary size.
    INCLUDE_REFLECTION_ATTRS = 1 << 0u,
  };

  // iree_hal_executable_library_version_t
  enum class Version : uint32_t {
    // IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0
    V_0 = 0u,
  };

  // iree_hal_executable_library_features_t
  enum class Features : uint32_t {
    // IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE
    NONE = 0u,
  };

  // iree_hal_executable_library_sanitizer_kind_t
  enum class SanitizerKind : uint32_t {
    // IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE
    NONE = 0u,
    // IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_ADDRESS
    ADDRESS = 1u,
    // IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_MEMORY
    MEMORY = 2u,
    // IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_THREAD
    THREAD = 3u,
    // IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_UNDEFINED
    UNDEFINED = 4u,
  };

  LibraryBuilder(llvm::Module *module, Mode mode,
                 Version version = Version::V_0)
      : module(module), mode(mode), version(version) {}

  // Adds a new required feature flag bit. The runtime must support the feature
  // for the library to be usable.
  void addRequiredFeature(Features feature) {
    features = static_cast<Features>(static_cast<uint32_t>(features) |
                                     static_cast<uint32_t>(feature));
  }

  // Sets the LLVM sanitizer the executable is built to be used with. The
  // runtime must also be compiled with the same sanitizer enabled.
  void setSanitizerKind(SanitizerKind sanitizerKind) {
    this->sanitizerKind = sanitizerKind;
  }

  // Defines a new entry point on the library implemented by |func|.
  // |name| will be used as the library export and an optional |tag| will be
  // attached.
  void addEntryPoint(StringRef name, StringRef tag, llvm::Function *func) {
    entryPoints.push_back({name.str(), tag.str(), func});
  }

  // Builds a `iree_hal_executable_library_query_fn_t` with the given
  // |queryFuncName| that will return the current library metadata.
  //
  // The returned function will be inserted into the module with internal
  // linkage. Callers may change the linkage based on their needs (such as
  // exporting if producing a dynamic library or making dso_local for static
  // libraries that reference the function via extern in another compilation
  // unit, etc).
  llvm::Function *build(StringRef queryFuncName);

 private:
  // Builds and returns an iree_hal_executable_library_v0_t global constant.
  llvm::Constant *buildLibraryV0(std::string libraryName);

  llvm::Module *module = nullptr;
  Mode mode = Mode::INCLUDE_REFLECTION_ATTRS;
  Version version = Version::V_0;
  Features features = Features::NONE;
  SanitizerKind sanitizerKind = SanitizerKind::NONE;

  struct EntryPoint {
    std::string name;
    std::string tag;
    llvm::Function *func;
  };
  std::vector<EntryPoint> entryPoints;
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LIBRARYBUILDER_H_
