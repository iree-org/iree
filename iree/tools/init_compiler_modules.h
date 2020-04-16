// Copyright 2020 Google LLC
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

#ifndef IREE_TOOLS_INIT_COMPILER_MODULES_H_
#define IREE_TOOLS_INIT_COMPILER_MODULES_H_

#include "iree/compiler/Dialect/Modules/Check/IR/CheckDialect.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Dialect.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListDialect.h"

namespace mlir {
namespace iree_compiler {

// This function should be called before creating any MLIRContext if one expect
// all the possible dialects to be made available to the context automatically.
inline void registerIreeCompilerModuleDialects() {
  static bool init_once = []() {
    registerDialect<IREE::Check::CheckDialect>();
    registerDialect<IREE::Strings::StringsDialect>();
    registerDialect<IREE::TensorList::TensorListDialect>();
    return true;
  }();
  (void)init_once;
}
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_COMPILER_MODULES_H_
