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

#ifndef IREE_DIALECT_SHAPE_BASE
#define IREE_DIALECT_SHAPE_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Shape dialect.
//===----------------------------------------------------------------------===//

// TODO(b/143787186): rename when old dialects are removed.
def Shape_Dialect : Dialect {
  let name = "shape";
  let cppNamespace = "Shape";

  let summary = [{
    A dialect of helper ops for shapifying computations.
  }];
}

//===----------------------------------------------------------------------===//
// General types and helpers
//===----------------------------------------------------------------------===//

def Shape_RankedShape :
    Type<CPred<"$_self.isa<::mlir::iree_compiler::Shape::RankedShapeType>()">,
         "Ranked shape type">;

#endif  // IREE_DIALECT_SHAPE_BASE
