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

class IREE_ScalarMemRefOf<list<Type> allowedTypes> :
    MemRefRankOf<allowedTypes, [0]>;

//===----------------------------------------------------------------------===//
// High-level types
//===----------------------------------------------------------------------===//

def IREEHL_Bool :
    AnyTypeOf<[I1, I8], "boolean-storing type (1 or 8 -bit integer)">;
def IREEHL_Element : AnyTypeOf<[AnyInteger, AnyFloat]>;

def IREEHL_MemRef : MemRefOf<[IREEHL_Element]>;
def IREEHL_BoolMemRef : MemRefOf<[IREEHL_Bool]>;
def IREEHL_IntMemRef : MemRefOf<[AnyInteger]>;
def IREEHL_FloatMemRef : MemRefOf<[AnyFloat]>;
def IREEHL_IndexMemRef : MemRefOf<[AnyInteger]>;

def IREEHL_AnyScalar : IREE_ScalarMemRefOf<[IREEHL_Element]>;
def IREEHL_BoolScalar : IREE_ScalarMemRefOf<[IREEHL_Bool]>;
def IREEHL_IntScalar : IREE_ScalarMemRefOf<[AnyInteger]>;
def IREEHL_FloatScalar : IREE_ScalarMemRefOf<[AnyFloat]>;
def IREEHL_IndexScalar : IREE_ScalarMemRefOf<[AnyInteger]>;
def IREEHL_I32Scalar : IREE_ScalarMemRefOf<[I32]>;

def IREEHL_1DIntMemRef : MemRefRankOf<[AnyInteger], [1]>;
def IREEHL_1DIndexMemRef : MemRefRankOf<[AnyInteger], [1]>;


//===----------------------------------------------------------------------===//
// Low-level types
//===----------------------------------------------------------------------===//

def IREELL_Bool : TypeAlias<I8, "boolean-storing type (8-bit integer)">;
def IREELL_Int: AnyTypeOf<[I8, I16, I32, I64], "8/16/32/64-bit integer">;
def IREELL_Float: AnyTypeOf<[F32, F64], "32/64-bit float">;
def IREELL_Index : AnyTypeOf<[I32, I64], "32/64-bit index integer">;
def IREELL_Element : AnyTypeOf<[IREELL_Int, IREELL_Float]>;

def IREELL_MemRef : MemRefOf<[IREELL_Element]>;
def IREELL_IntMemRef : MemRefOf<[IREELL_Int]>;
def IREELL_FloatMemRef : MemRefOf<[IREELL_Float]>;
def IREELL_BoolMemRef : MemRefOf<[IREELL_Bool]>;
def IREELL_IndexMemRef : MemRefOf<[IREELL_Index]>;
// For shape computation outputs, we want to consistently output I32 not I64
// TODO(b/138851470) Iron out story for index types
def IREELL_I32MemRef : MemRefOf<[I32]>;

def IREELL_ElementScalar : IREE_ScalarMemRefOf<[IREELL_Element]>;
def IREELL_IntScalar : IREE_ScalarMemRefOf<[IREELL_Int]>;
def IREELL_BoolScalar : IREE_ScalarMemRefOf<[IREELL_Bool]>;
def IREELL_FloatScalar : IREE_ScalarMemRefOf<[IREELL_Float]>;
def IREELL_IndexScalar : IREE_ScalarMemRefOf<[IREELL_Index]>;
// For shape computation outputs, we want to consistently output I32 not I64
// TODO(b/138851470) Iron out story for index types
def IREELL_I32Scalar : IREE_ScalarMemRefOf<[I32]>;

def IREELL_1DIntMemRef : MemRefRankOf<[IREELL_Int], [1]>;
def IREELL_1DIndexMemRef : MemRefRankOf<[IREELL_Index], [1]>;

#endif  // IREE_OP_BASE
