// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_ENCODING_OPS
#define IREE_DIALECT_ENCODING_OPS

include "iree/compiler/Dialect/Encoding/IR/EncodingBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class IREEEncoding_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<IREEEncoding_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// set_encoding op.
//===----------------------------------------------------------------------===//

def IREEEncoding_SetEncodingOp : IREEEncoding_PureOp<"set_encoding",[
   DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>, Pure
  ]> {
  let summary = "perform pack and pad operation on source";
  let description = [{
    Operation to assign an encoding to a tensor. The operation
    does not change the rank or extent of a tensor. Instead it
    adds an encoding attribute to the tensor type to represent
    a change in layout.
  }];

  let arguments = (ins AnyRankedTensor:$source);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    attr-dict $source `:` type($source) `->` type($result)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    RankedTensorType getSourceType() {
      return cast<RankedTensorType>(getSource().getType());
    }
    RankedTensorType getResultType() {
      return cast<RankedTensorType>(getResult().getType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// unset_encoding op.
//===----------------------------------------------------------------------===//

def IREEEncoding_UnsetEncodingOp : IREEEncoding_PureOp<"unset_encoding", [
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>, Pure
  ]> {
  let summary = "perfom unpack and extract operation on source";
  let description = [{
    Operation to convert an tensor with encoding that represents
    its data layout into a tensor with default layout (i.e. no encoding).
    For now in IREE the default layout is row-major.
  }];
  let arguments = (ins AnyRankedTensor:$source);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    attr-dict $source `:` type($source) `->` type($result)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    RankedTensorType getSourceType() {
      return cast<RankedTensorType>(getSource().getType());
    }
    RankedTensorType getResultType() {
      return cast<RankedTensorType>(getResult().getType());
    }
  }];
}

#endif  // IREE_DIALECT_ENCODING_OPS
