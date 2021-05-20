// Copyright 2021 Google LLC
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

#ifndef IREE_COMPILER_CONVERSION_LINALGTOLLVM_FOLDTENSOREXTRACTOP
#define IREE_COMPILER_CONVERSION_LINALGTOLLVM_FOLDTENSOREXTRACTOP

include "mlir/Dialect/MemRef/IR/MemRefOps.td"
include "mlir/Dialect/Tensor/IR/TensorOps.td"

// Canonicalize unnecessary tensor_load when the load is used just for
// an extract
def : Pat<(Tensor_ExtractOp (TensorLoadOp $value), $indices),
          (LoadOp $value, $indices)>;

#endif // IREE_COMPILER_CONVERSION_LINALGTOLLVM_FOLDTENSOREXTRACTOP
