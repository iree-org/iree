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

#ifndef IREE_COMPILER_DIALECT_VM_IR_VMTYPES_H_
#define IREE_COMPILER_DIALECT_VM_IR_VMTYPES_H_

#include "iree/compiler/Dialect/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "third_party/llvm/llvm/include/llvm/ADT/DenseMapInfo.h"
#include "third_party/llvm/llvm/include/llvm/ADT/SmallVector.h"
#include "third_party/llvm/llvm/include/llvm/ADT/StringSwitch.h"

// Order matters.
#include "iree/compiler/Dialect/VM/IR/VMEnums.h.inc"

#endif  // IREE_COMPILER_DIALECT_VM_IR_VMTYPES_H_
