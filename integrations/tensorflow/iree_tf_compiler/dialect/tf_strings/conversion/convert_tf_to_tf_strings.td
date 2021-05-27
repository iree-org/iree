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

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "iree_tf_compiler/dialect/tf_strings/ir/ops.td"

def : Pat<(TF_IdentityOp TF_StrTensor:$value),
           (replaceWithValue $value)>;

def : Pat<(TF_PrintV2Op TF_Str:$value, $unused1, $unused2),
          (TFStrings_PrintOp $value)>;

def : Pat<(TF_PrintV2Op TF_StrTensor:$value, $unused1, $unused2),
          (TFStrings_PrintOp $value)>;

def : Pat<(TF_AsStringOp $value, $u1, $u2, $u3, $u4, $u5),
          (TFStrings_ToStringTensorOp $value)>;
