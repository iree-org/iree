// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_HAL_LOADER_OPS
#define IREE_DIALECT_MODULES_HAL_LOADER_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderBase.td"
include "iree/compiler/Dialect/Util/IR/UtilAttrs.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class HALLoader_PureOp<string mnemonic, list<Trait> traits = []> :
    HALLoader_Op<mnemonic, !listconcat(traits, [Pure])>;

//===----------------------------------------------------------------------===//
// !hal.executable / iree_hal_executable_t
//===----------------------------------------------------------------------===//

def OpGroupExecutableOps : OpDocGroup {
  let summary = "Executable ops";
  let description = "Ops for `!hal.executable` / `iree_hal_executable_t`.";
}

let opDocGroup = OpGroupExecutableOps in {

def HALLoader_ExecutableQuerySupportOp : HALLoader_PureOp<"executable.query_support", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = "queries whether an executable format is supported";
  let description = [{
    Returns true if the given format is supported by the device loader. This
    does not guarantee that loading will succeed as the executable may require
    functionality that cannot be met my the hosting runtime environment.
  }];

  let arguments = (ins
    StrAttr:$executable_format
  );
  let results = (outs
    I1:$supported
  );

  let assemblyFormat = [{
    `format` `(` $executable_format `)`
    `:` type($supported)
    attr-dict-with-keyword
  }];
}

def HALLoader_ExecutableLoadOp : HALLoader_PureOp<"executable.load", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = "dynamically loads an executable";
  let description = [{
    Creates, loads, and dynamically links an executable.

    Optional constants provide for specialization of the executable based on
    runtime-derived parameters.
  }];

  let arguments = (ins
    StrAttr:$format,
    Util_BufferType:$data,
    Variadic<I32>:$constants
  );
  let results = (outs
    HAL_Executable:$result
  );

  let assemblyFormat = [{
    `format` `(` $format `)`
    `data` `(` $data `)`
    (`constants` `(` `[` $constants^ `]` `)`)?
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HALLoader_ExecutableLookupOp : HALLoader_PureOp<"executable.lookup", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = [{executable cache lookup pseudo-op}];
  let description = [{
    Used during conversion to provide a placeholder for a globally cached and
    possibly lazy-initialized executable.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$executable
  );
  let results = (outs
    HAL_Executable:$result
  );

  let assemblyFormat = [{
    `executable` `(` $executable `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HALLoader_ExecutableExportOrdinalOp : HALLoader_PureOp<"executable.export.ordinal", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = [{executable export ordinal lookup pseudo-op}];
  let description = [{
    Resolves an executable export ordinal to a value once ordinals have been
    assigned.
  }];

  let arguments = (ins
    SymbolRefAttr:$entry_point
  );
  let results = (outs
    Index:$result
  );

  let assemblyFormat = [{
    `target` `(` $entry_point `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HALLoader_ExecutableDispatchOp : HALLoader_Op<"executable.dispatch", [
  AttrSizedOperandSegments,
]> {
  let summary = [{inline executable dispatch operation}];
  let description = [{
    Dispatches execution to an executable entry point with the given parameters.
  }];

  let arguments = (ins
    HAL_Executable:$executable,
    HAL_Ordinal:$entry_point,
    HAL_Dim:$workgroup_x,
    HAL_Dim:$workgroup_y,
    HAL_Dim:$workgroup_z,
    Variadic<I32>:$push_constants,
    Variadic<Util_BufferType>:$binding_buffers,
    Variadic<HAL_DeviceSize>:$binding_offsets,
    Variadic<HAL_DeviceSize>:$binding_lengths
  );

  let assemblyFormat = [{
    `executable` `(` $executable `:` type($executable) `)`
    `` `[` $entry_point `]`
    `workgroups` `(` `[`
        $workgroup_x `,`
        $workgroup_y `,`
        $workgroup_z
    `]` `)`
    (`constants` `(` `[` $push_constants^ `]` `)`)?
    `bindings` `(` `[`
    custom<DispatchBindings>($binding_buffers,
                             type($binding_buffers),
                             $binding_offsets,
                             $binding_lengths)
    `]` `)`
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

} // OpGroupExecutableOps

#endif  // IREE_DIALECT_MODULES_HAL_LOADER_OPS
