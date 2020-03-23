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

#ifndef IREE_INDEX_COMPUTATION_ATTR
#define IREE_INDEX_COMPUTATION_ATTR

include "mlir/Dialect/Affine/IR/AffineOpsBase.td"
include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

// TODO(ravishankarm) : Using this in tablegen returns a compiler
// error in the generated CPP
// def OperandIndicesAttr : TypedArrayAttrBase<AffineMapArrayAttr,
//                                             "Operand indices attribute">;

// This attribute captures the result of index propogation for each op and for
// arguments.
// 1) The `result_index` field contains a list of affine maps that describe the
//    location of the result tensors accessed within a workitem (or thread). The
//    size of the list is same as the number of results produced by the
//    operation.
// 2) The index propagation computes the location of the operand tensors
//    needed to compute the value at a location of the result tensor. For
//    each operand, values from more than one location might be needed for
//    computing the result (for ex. xla_hlo.gather, xla_hlo.reduce_window,
//    etc). The indices needed for each operand are stored as an ArrayAttr
//    of AffineMaps. `operand_indices` is a list of these ArrayAttr, with
//    one entry for each operand. So `operand_indices` is really
//    ArrayAttr<AffineMapArrayAttr>
// Note that for Value (either a result of an operation, or an argument),
// multiple indices might be needed within a workitem. So each
// operation/argument will be decorated withn an ArrayAttr, which is a
// list of `IndexAttr`, after index propagation pass.
def IndexComputationAttr :
    StructAttr<"IndexAttr", IREE_Dialect,
               [StructFieldAttr<"result_index", AffineMapArrayAttr>,
                StructFieldAttr<"operand_indices", ArrayAttr>]>;

#endif // IREE_INDEX_COMPUTATION_ATTR
