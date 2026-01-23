// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_BASE
#define IREE_DIALECT_HAL_BASE

include "iree/compiler/Dialect/HAL/IR/HALDialect.td"
include "iree/compiler/Dialect/HAL/IR/HALInterfaces.td"
include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.td"
include "iree/compiler/Utils/DocMetadata.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// HAL traits
//===----------------------------------------------------------------------===//

// Denotes that an operation is part of the executable interface.
def HAL_ExecutableInterfaceOp : NativeOpTrait<"IREE::HAL::ExecutableInterfaceOp">;

//===----------------------------------------------------------------------===//
// Base dialect op/type/attr/interface classes
//===----------------------------------------------------------------------===//

class HAL_Op<string mnemonic, list<Trait> traits = []> :
    Op<HAL_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;

  // Optional documentation metadata for rich doc generation.
  Util_OpDocMetadata docMetadata = ?;
}

class HAL_PureOp<string mnemonic, list<Trait> traits = []> :
    HAL_Op<mnemonic, !listconcat(traits, [Pure])>;

class HAL_MakeTupleOp<string mnemonic, list<Trait> traits = []> :
    HAL_PureOp<mnemonic, traits>;

class HAL_TypeDef<string name, list<Trait> traits = []> :
    TypeDef<HAL_Dialect, name, traits> {
  // Optional documentation metadata for rich doc generation.
  Util_TypeDocMetadata docMetadata = ?;
}

class HAL_AttrDef<string name, list<Trait> traits = []> :
    AttrDef<HAL_Dialect, name, traits> {
  // Optional documentation metadata for rich doc generation.
  Util_AttrDocMetadata docMetadata = ?;
}

class HAL_OpInterface<string name, list<Interface> baseInterfaces = []> :
    OpInterface<name, baseInterfaces> {
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

class HAL_TypeInterface<string name, list<Interface> baseInterfaces = []> :
    TypeInterface<name, baseInterfaces> {
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

class HAL_AttrInterface<string name, list<Interface> baseInterfaces = []> :
    AttrInterface<name, baseInterfaces> {
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

#endif  // IREE_DIALECT_HAL_BASE
