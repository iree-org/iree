// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef STRINGS_BASE
#define STRINGS_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE execution flow dialect
//===----------------------------------------------------------------------===//

def TFStrings_Dialect : Dialect {
  let name = "tf_strings";
  let cppNamespace = "::mlir::iree_integrations::tf_strings";
  let summary = [{
    A strings dialect for simplifying tensorflow string operations.
  }];
  let description = [{
    A strings dialect for simplifying tensorflow string operations.
  }];

}

def TFStrings_String : DialectType<
    TFStrings_Dialect,
    CPred<"$_self.isa<tf_strings::StringType>()">,
    "string"> {
  let description = [{
    String type representation.
  }];
}

class TFStrings_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TFStrings_Dialect, mnemonic, traits> {
}


def TFStrings_Value : AnyTypeOf<[AnyFloat, AnySignlessInteger]>;
def TFStrings_StringTensor : TensorOf<[TFStrings_String]>;
def TFStrings_ValueTensor : TensorOf<[AnyFloat, AnySignlessInteger]>;

#endif //  STRINGS_BASE
