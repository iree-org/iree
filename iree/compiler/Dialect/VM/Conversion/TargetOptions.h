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

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_TARGETOPTIONS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_TARGETOPTIONS_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// Defines runtime VM extension opcode sets.
enum class OpcodeExtension {
  // Adds ops for manipulating i64 types.
  kI64,
};

// Controls VM translation targets.
struct TargetOptions {
  // Target size of `index` when converted to an integer in bits.
  int indexBits = 32;

  // Whether the i64 extension is enabled in the target VM.
  bool i64Extension = false;

  // Whether to truncate i64 types to i32 when the i64 extension is not
  // enabled.
  bool truncateUnsupportedIntegers = true;
};

// Returns a TargetOptions struct initialized with the
// --iree-vm-target-* flags.
TargetOptions getTargetOptionsFromFlags();

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_TARGETOPTIONS_H_
