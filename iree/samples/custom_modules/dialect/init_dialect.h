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

// This files defines a helper to trigger the registration of the custom
// dialect to the system.
//
// Imported from the iree-template-cpp project.

#ifndef IREE_SAMPLES_CUSTOM_MODULES_DIALECT_INIT_DIALECT_H_
#define IREE_SAMPLES_CUSTOM_MODULES_DIALECT_INIT_DIALECT_H_

#include "iree/samples/custom_modules/dialect/custom_dialect.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_compiler {

// Add custom dialect to the provided registry.
inline void registerCustomDialect(DialectRegistry &registry) {
  registry.insert<IREE::Custom::CustomDialect>();
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_SAMPLES_CUSTOM_MODULES_DIALECT_INIT_DIALECT_H_
