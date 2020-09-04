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

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_TYPECONVERTER_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_TYPECONVERTER_H_

#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

class TypeConverter : public mlir::TypeConverter {
 public:
  explicit TypeConverter(
      TargetOptions targetOptions = getTargetOptionsFromFlags());

  const TargetOptions& targetOptions() const { return targetOptions_; }

 private:
  TargetOptions targetOptions_;
};

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_TYPECONVERTER_H_
