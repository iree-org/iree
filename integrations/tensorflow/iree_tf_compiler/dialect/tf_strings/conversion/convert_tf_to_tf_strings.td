// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
