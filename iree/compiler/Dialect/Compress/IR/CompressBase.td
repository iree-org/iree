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

#ifndef IREE_DIALECT_COMPRESS_BASE
#define IREE_DIALECT_COMPRESS_BASE

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

//===----------------------------------------------------------------------===//
// IREE Compress dialect
//===----------------------------------------------------------------------===//

def Compress_Dialect : Dialect {
  let name = "iree_compress";
  let cppNamespace = "IREE::Compress";

  let summary = [{
    A dialect of ops for representing computation compression transformations.
  }];
  let description = [{
    This dialect brings together quantization tooling and transforms for
    preparing computations for lower precision code generation.
  }];
}

class Compress_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Compress_Dialect, mnemonic, traits> {
}

#endif  // IREE_DIALECT_COMPRESS_BASE
