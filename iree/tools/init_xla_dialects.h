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

// This files defines a helper to trigger the registration of dialects to
// the system.

#ifndef IREE_TOOLS_INIT_XLA_DIALECTS_H_
#define IREE_TOOLS_INIT_XLA_DIALECTS_H_

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir/IR/Dialect.h"

namespace mlir {

// Add all the XLA dialects to the provided registry.
inline void registerXLADialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::chlo::HloClientDialect,
                  mlir::lmhlo::LmhloDialect,
                  mlir::mhlo::MhloDialect>();
  // clang-format on
}

}  // namespace mlir

#endif  // IREE_TOOLS_INIT_XLA_DIALECTS_H_
