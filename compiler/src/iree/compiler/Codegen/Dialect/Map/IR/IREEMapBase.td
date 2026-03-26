// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_MAP_IREEMAP_BASE
#define IREE_CODEGEN_DIALECT_MAP_IREEMAP_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// IREE Map Dialect
//===----------------------------------------------------------------------===//

def IREEMap_Dialect : Dialect {
  let name = "iree_map";
  let cppNamespace = "::mlir::iree_compiler::IREE::Map";

  let summary = [{
    A dialect for map algebra used in code generation.
  }];
  let description = [{
    This dialect provides map attributes for describing hierarchical coordinate
    mappings used in codegen. The base of the dialect is the PackMap attribute
    which defines a coordinate to index mapping and defines a closed algebra
    over it. This algebra allows us to represent complex coordinate
    transformations as algebra operations.
  }];

  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Attribute base class
//===----------------------------------------------------------------------===//

class IREEMap_Attr<string name, list<Trait> traits = []>
  : AttrDef<IREEMap_Dialect, name, traits>;

#endif // IREE_CODEGEN_DIALECT_MAP_IREEMAP_BASE
