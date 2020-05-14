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

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"

#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {

bool isEntryPoint(FuncOp func) {
  return SymbolTable::getSymbolVisibility(func) ==
         SymbolTable::Visibility::Public;
}

}  // namespace iree_compiler
}  // namespace mlir
