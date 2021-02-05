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

// This file defines a helper to add passes to the global registry.

#ifndef IREE_TOOLS_INIT_PASSES_H_
#define IREE_TOOLS_INIT_PASSES_H_

#include <cstdlib>

#include "iree/compiler/Conversion/init_conversions.h"
#include "iree/compiler/Dialect/HAL/Conversion/Passes.h"
#include "iree/tools/init_iree_passes.h"
#include "iree/tools/init_mlir_passes.h"

namespace mlir {
namespace iree_compiler {

// Registers IREE core passes and other important passes to the global registry.
inline void registerAllPasses() {
  registerAllIreePasses();
  registerCommonConversionPasses();
  registerMlirPasses();
  registerHALConversionPasses();
  registerLinalgToSPIRVPasses();
  registerHLOToLinalgPasses();
  registerLinalgToLLVMPasses();
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_PASSES_H_
