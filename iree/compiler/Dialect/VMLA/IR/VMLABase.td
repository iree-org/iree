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

#ifndef IREE_DIALECT_VMLA_BASE
#define IREE_DIALECT_VMLA_BASE

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

//===----------------------------------------------------------------------===//
// IREE VMLA (Virtual Machine-based Linear Algebra) dialect
//===----------------------------------------------------------------------===//

def VMLA_Dialect : Dialect {
  let name = "vmla";
  let cppNamespace = "IREE::VMLA";

  let summary = [{
    A dialect representing operations against the IREE VM-based backend.
  }];
  let description = [{
    This is a reference dialect representing a simple IREE VM-based linear
    algebra module that is used as a library at runtime. The ops in this dialect
    map (roughly) 1:1 with the exported functions in the runtime module.

    See `vmla.imports.mlir` for the full list of exported functions.
  }];
}

//===----------------------------------------------------------------------===//
// VMLA enums
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VMLA types
//===----------------------------------------------------------------------===//

def VMLA_DeviceSize : TypeAlias<I32>;
def VMLA_DeviceSizeAttr : IntegerAttrBase<I32, "device_size_t">;

def VMLA_HostSize : TypeAlias<I32>;
def VMLA_HostSizeAttr : IntegerAttrBase<I32, "size_t">;

def VMLA_Dim : I<32>;
def VMLA_Dims : Variadic<VMLA_Dim>;
def VMLA_Shape : Variadic<VMLA_Dim>;

def HAL_HostBufferRef : AnyTypeOf<[
  RefPtrOf<ByteBufferType>,
  RefPtrOf<MutableByteBufferType>,
]>;

//===----------------------------------------------------------------------===//
// Base VMLA op classes
//===----------------------------------------------------------------------===//

def VMLA_OpInterface : OpInterface<"VMLAOp"> {
  let description = [{
    Interface for VMLA ops.
  }];
}

class VMLA_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<VMLA_Dialect, mnemonic, !listconcat(traits, [VMLA_OpInterface])> {
  // TODO(benvanik): use new tablegen printer/parser.
  // let parser = [{ return parse$cppClass(parser, &result); }];
  // let printer = [{ return print$cppClass(p, *this); }];
}

class VMLA_PureOp<string mnemonic, list<OpTrait> traits = []> :
    VMLA_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

#endif  // IREE_DIALECT_VMLA_BASE
