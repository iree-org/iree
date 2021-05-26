// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_BASE_TD
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_BASE_TD

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

def TFTensorList_Dialect : Dialect {
  let name = "tf_tensorlist";

  let summary = "Dialect used to represent TensorFlow TensorList's in MLIR.";

  let description = [{
    This is a "frontend" dialect intended to map 1:1 to TensorFlow's TensorList
    ops. The main difference from the `tf` dialect is that this dialect
    has a dedicated TensorList type, instead of using a `tensor<!tf.variant>`
    to represent TensorLists.

    This allows representing the TensorList construct with greater fidelity.
  }];

  let cppNamespace = "::mlir::iree_integrations::tf_tensorlist";
}

def TFTensorList_TensorList :
    Type<CPred<"$_self.isa<mlir::iree_integrations::tf_tensorlist::TensorListType>()">, "list">;

#endif  // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_BASE_TD
