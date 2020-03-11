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

#ifndef IREE_COMPILER_DIALECT_VMLA_IR_VMLAOPS_H_
#define IREE_COMPILER_DIALECT_VMLA_IR_VMLAOPS_H_

#include <cstdint>

#include "iree/compiler/Dialect/IREE/IR/IREETraits.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATraits.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffects.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMLA {

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h.inc"

}  // namespace VMLA
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VMLA_IR_VMLAOPS_H_
