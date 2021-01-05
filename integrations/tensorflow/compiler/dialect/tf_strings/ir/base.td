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
