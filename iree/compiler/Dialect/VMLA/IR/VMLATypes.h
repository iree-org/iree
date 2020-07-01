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

#ifndef IREE_COMPILER_DIALECT_VMLA_IR_VMLATYPES_H_
#define IREE_COMPILER_DIALECT_VMLA_IR_VMLATYPES_H_

#include <cstdint>

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// Order matters.
#include "iree/compiler/Dialect/VMLA/IR/VMLAEnums.h.inc"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMLA {

#include "iree/compiler/Dialect/VMLA/IR/VMLAOpInterface.h.inc"

//===----------------------------------------------------------------------===//
// RefObject types
//===----------------------------------------------------------------------===//

class BufferType : public Type::TypeBase<BufferType, Type, TypeStorage> {
 public:
  using Base::Base;
  static BufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Buffer);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::Buffer; }
};

class InterfaceType : public Type::TypeBase<InterfaceType, Type, TypeStorage> {
 public:
  using Base::Base;
  static InterfaceType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Interface);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::Interface; }
};

}  // namespace VMLA
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VMLA_IR_VMLATYPES_H_
