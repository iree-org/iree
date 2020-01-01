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

// Common IREE op definitions shared by the interpreter and sequencer dialects.

#ifndef IREE_OP_BASE
#define IREE_OP_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Top-level IREE dialect
//===----------------------------------------------------------------------===//

def IREE_Dialect : Dialect {
  let name = "iree";
  let cppNamespace = "IREE";
}

//===----------------------------------------------------------------------===//
// General types and helpers
//===----------------------------------------------------------------------===//

def IREE_Bool :
    AnyTypeOf<[I1, I8], "boolean-storing type (1 or 8 -bit integer)">;
def IREE_Element : AnyTypeOf<[AnyInteger, AnyFloat]>;
def IREE_MemRef : MemRefOf<[IREE_Element]>;

#endif  // IREE_OP_BASE
