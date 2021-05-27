// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_TENSORLIST_BASE
#define IREE_DIALECT_MODULES_TENSORLIST_BASE

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

def TensorList_Dialect : Dialect {
  let name = "tensorlist";
  let cppNamespace = "::mlir::iree_compiler::IREE::TensorList";

  let summary = [{
    A dialect for implementing TensorFlow's TensorList data structure.
  }];
  let description = [{
    This dialect contains ops that map 1:1 to the corresponding IREE native
    module.

    This dialect mimics TensorFlow's TensorList op set. It provides a
    a straightforward implementation of those semantics.
  }];
}

def TensorList_TensorList : DialectType<
    TensorList_Dialect,
    CPred<"$_self.isa<IREE::TensorList::TensorListType>()">,
    "tensorlist.list"> {
  let description = [{
    IREE VM type representing a TensorList.
  }];
}


#endif  // IREE_DIALECT_MODULES_TENSORLIST_BASE
