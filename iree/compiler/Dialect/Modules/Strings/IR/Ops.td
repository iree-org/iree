// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_STRINGS_MODULES_DIALECT_STRINGS_OPS
#define IREE_SAMPLES_STRINGS_MODULES_DIALECT_STRINGS_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/IREE/IR/IREEBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def STRINGS_ValueTensor : AnyTypeOf<[HAL_BufferView, AnyRankedTensor]>;

def STRINGS_Dialect : Dialect {
  let name = "strings";
  let cppNamespace = "::mlir::iree_compiler::IREE::Strings";

  let summary = [{
    A custom dialect supporting string operations.
  }];
  let description = [{
    The ops in this dialect are lowered to vm.imports to support a standard set
    of string operations.
    TODO(suderman): Add a reference to the intermediate string dialect.
  }];
}

def STRINGS_String : DialectType<
    STRINGS_Dialect,
    CPred<"$_self.isa<IREE::Strings::StringType>()">,
    "string"> {
  let description = [{
    A string type containing a standard ASCII string.
  }];
}

def STRINGS_StringTensor : DialectType<
    STRINGS_Dialect,
    CPred<"$_self.isa<IREE::Strings::StringTensorType>()">,
    "string_tensor"> {
  let description = [{
    A tensor of string types.
  }];
}

def STRINGS_I32ToStringOp : Op<STRINGS_Dialect, "i32_to_string", [NoSideEffect]> {
  let summary = [{converts an i32 to a string}];
  let description = [{
    Converts an i32 to its string representation.
  }];

  let arguments = (ins I32:$value);

  let results = (outs
    STRINGS_String:$result
  );
}

def STRINGS_ToStringTensorOp : Op<STRINGS_Dialect, "to_string_tensor", [NoSideEffect]> {
  let summary = [{converts a hal buffer to a string tensor}];
  let description = [{
    Converts a hal buffer to a tensor of strings representation.
  }];

  let arguments = (ins STRINGS_ValueTensor:$value);

  let results = (outs
    STRINGS_StringTensor:$result
  );
}

def STRINGS_StringTensorToStringOp : Op<STRINGS_Dialect, "string_tensor_to_string", [NoSideEffect]> {
  let summary = [{converts a string tensor to a string}];
  let description = [{
    Converts a string tensor to a string.
  }];

  let arguments = (ins STRINGS_StringTensor:$value);

  let results = (outs
    STRINGS_String:$result
  );
}

def STRINGS_GatherOp : Op<STRINGS_Dialect, "gather", [NoSideEffect]> {
  let summary = "Gathers all the strings from a Tensor by index.";
  let description = [{
    Gathers all the strings from a Tensor by index values along the final axis.
  }];

  let arguments = (ins STRINGS_StringTensor:$dict,
                  STRINGS_ValueTensor:$indices);

  let results = (outs
    STRINGS_StringTensor:$result
  );
}

def STRINGS_ConcatOp : Op<STRINGS_Dialect, "concat", [NoSideEffect]> {
  let summary = "Concatenates a tensor of strings along the last dimension";

  let description = [{
    Concatenates the strings in the tensor along the last dimension.
  }];

  let arguments = (ins STRINGS_StringTensor:$value);

  let results = (outs
    STRINGS_StringTensor:$result
  );
}

def STRINGS_PrintOp : Op<STRINGS_Dialect, "print"> {
  let summary = "Prints the contents of a string.";
  let description = [{
    Prints the contents of a string.
  }];

  let arguments = (ins STRINGS_String:$value);
}

#endif  // IREE_SAMPLES_STRINGS_MODULES_DIALECT_STRINGS_OPS
