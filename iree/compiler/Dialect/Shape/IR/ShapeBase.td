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
  let name = "shapex";
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

// TODO(silvasean): Investigate the layering aspects of allowing non-index types
// here. There seem to be two primary motivators right now, both of which are
// not obviously ideal long-term:
//
// 1. mhlo dialect uses i64 in many places that index should be used.
//    This is understood to be a bug.
// 2. VMLA gets these values as i32 directly.
//
// Both cases can be resolved by inserting casts if we decide to take a firmer
// stance on only allowing index type. But retaining the flexibility might
// be a useful feature.
def Shape_DimType : AnyTypeOf<[Index, AnySignlessInteger]>;

def Shape_ExtentTensor : ShapedContainerType<
    [Index, AnySignlessInteger],
    And<[IsTensorTypePred, HasAnyRankOfPred<[1]>]>,
    "a 1D tensor of extents">;

#endif  // IREE_DIALECT_SHAPE_BASE
