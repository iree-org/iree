// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_IO_PARAMETERS_OPS
#define IREE_DIALECT_MODULES_IO_PARAMETERS_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/Util/IR/UtilAttrs.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "iree/compiler/Modules/IO/Parameters/IR/IOParametersBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IOParameters_PureOp<string mnemonic, list<Trait> traits = []> :
    IOParameters_Op<mnemonic, !listconcat(traits, [Pure])>;

//===----------------------------------------------------------------------===//
// Parameter I/O
//===----------------------------------------------------------------------===//

def OpGroupParameterOps : OpDocGroup {
  let summary = "Parameter I/O ops";
  let description = "Ops parameter I/O.";
}

let opDocGroup = OpGroupParameterOps in {

def IOParameters_LoadOp : IOParameters_Op<"load", [
  Util_SizeAwareOp,
]> {
  let summary = [{reads a parameter from a parameter scope}];
  let description = [{
    Asynchronously reads a parameter from an external parameter provider and
    returns the resulting buffer. Depending on the parameter and buffer types
    this may alias existing cached storage or be directly mapped to the
    parameter origin or result in a copy as if an allocate + read had been used.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DeviceQueueAffinity:$queue_affinity,
    HAL_Fence:$wait_fence,
    HAL_Fence:$signal_fence,
    OptionalAttr<StrAttr>:$source_scope,
    StrAttr:$source_key,
    I64:$source_offset,
    HAL_MemoryTypeBitfieldAttr:$memory_types,
    HAL_BufferUsageBitfieldAttr:$buffer_usage,
    HAL_DeviceSize:$length
  );
  let results = (outs
    HAL_Buffer:$result
  );

  let assemblyFormat = [{
    `<` $device `:` type($device) `>`
    `affinity` `(` $queue_affinity `)`
    `wait` `(` $wait_fence `)`
    `signal` `(` $signal_fence `)`
    `source` `(` custom<ParameterReference>($source_scope, $source_key) `)`
    `` `[` $source_offset `]`
    `type` `(` $memory_types `)`
    `usage` `(` $buffer_usage `)`
    `:` custom<SizeAwareType>(type($result), $length)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getLength(); }
  }];

  let hasCanonicalizer = 1;
}

def IOParameters_ReadOp : IOParameters_Op<"read", []> {
  let summary = [{reads a parameter from a parameter scope}];
  let description = [{
    Asynchronously reads a parameter from an external parameter provider into
    the provided target buffer range.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DeviceQueueAffinity:$queue_affinity,
    HAL_Fence:$wait_fence,
    HAL_Fence:$signal_fence,
    OptionalAttr<StrAttr>:$source_scope,
    StrAttr:$source_key,
    I64:$source_offset,
    HAL_Buffer:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length
  );

  let assemblyFormat = [{
    `<` $device `:` type($device) `>`
    `affinity` `(` $queue_affinity `)`
    `wait` `(` $wait_fence `)`
    `signal` `(` $signal_fence `)`
    `source` `(` custom<ParameterReference>($source_scope, $source_key) `)`
    `` `[` $source_offset `]`
    `target` `(` $target_buffer `:` type($target_buffer)  `)`
    `` `[` $target_offset `]`
    `length` `(` $length `)`
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def IOParameters_WriteOp : IOParameters_Op<"write", []> {
  let summary = [{writes a parameter into a parameter scope}];
  let description = [{
    Asynchronously writes a parameter to an external parameter provider from
    the provided source buffer range.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DeviceQueueAffinity:$queue_affinity,
    HAL_Fence:$wait_fence,
    HAL_Fence:$signal_fence,
    HAL_Buffer:$source_buffer,
    HAL_DeviceSize:$source_offset,
    OptionalAttr<StrAttr>:$target_scope,
    StrAttr:$target_key,
    I64:$target_offset,
    HAL_DeviceSize:$length
  );

  let assemblyFormat = [{
    `<` $device `:` type($device) `>`
    `affinity` `(` $queue_affinity `)`
    `wait` `(` $wait_fence `)`
    `signal` `(` $signal_fence `)`
    `source` `(` $source_buffer `:` type($source_buffer)  `)`
    `` `[` $source_offset `]`
    `target` `(` custom<ParameterReference>($target_scope, $target_key) `)`
    `` `[` $target_offset `]`
    `length` `(` $length `)`
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def IOParameters_GatherOp : IOParameters_Op<"gather", [
  AttrSizedOperandSegments,
]> {
  let summary = [{gathers multiple parameters from a parameter scope}];
  let description = [{
    Asynchronously gathers one or more parameters into a single target buffer.
    This is equivalent to one read per parameter but allows implementations that
    can batch operations to do so without additional overhead.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DeviceQueueAffinity:$queue_affinity,
    HAL_Fence:$wait_fence,
    HAL_Fence:$signal_fence,
    OptionalAttr<StrAttr>:$source_scope,
    StrArrayAttr:$source_keys,
    Variadic<I64>:$source_offsets,
    HAL_Buffer:$target_buffer,
    Variadic<HAL_DeviceSize>:$target_offsets,
    Variadic<HAL_DeviceSize>:$target_lengths
  );

  let assemblyFormat = [{
    `<` $device `:` type($device) `>`
    `affinity` `(` $queue_affinity `)`
    `wait` `(` $wait_fence `)`
    `signal` `(` $signal_fence `)`
    `{`
    custom<ParameterGatherOperations>(
        $source_scope, $source_keys, $source_offsets,
        $target_buffer, type($target_buffer), $target_offsets, $target_lengths)
    `}`
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def IOParameters_ScatterOp : IOParameters_Op<"scatter", [
  AttrSizedOperandSegments,
]> {
  let summary = [{scatters multiple parameters to a parameter scope}];
  let description = [{
    Asynchronously scatters one or more parameters from a single source buffer
    into one or more parameters. This is equivalent to one write per parameter
    but allows implementations that can batch operations to do so without
    additional overhead.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DeviceQueueAffinity:$queue_affinity,
    HAL_Fence:$wait_fence,
    HAL_Fence:$signal_fence,
    HAL_Buffer:$source_buffer,
    Variadic<HAL_DeviceSize>:$source_offsets,
    Variadic<HAL_DeviceSize>:$source_lengths,
    OptionalAttr<StrAttr>:$target_scope,
    StrArrayAttr:$target_keys,
    Variadic<I64>:$target_offsets
  );

  let assemblyFormat = [{
    `<` $device `:` type($device) `>`
    `affinity` `(` $queue_affinity `)`
    `wait` `(` $wait_fence `)`
    `signal` `(` $signal_fence `)`
    `{`
    custom<ParameterScatterOperations>(
        $source_buffer, type($source_buffer), $source_offsets, $source_lengths,
        $target_scope, $target_keys, $target_offsets)
    `}`
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

} // OpGroupParameterOps

#endif  // IREE_DIALECT_MODULES_IO_PARAMETERS_OPS
