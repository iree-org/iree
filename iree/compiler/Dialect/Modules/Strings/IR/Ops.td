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

#ifndef IREE_SAMPLES_STRINGS_MODULES_DIALECT_STRINGS_OPS
#define IREE_SAMPLES_STRINGS_MODULES_DIALECT_STRINGS_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/IREE/IR/IREEBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffects.td"

def STRINGS_Dialect : Dialect {
  let name = "strings";
  let cppNamespace = "IREE::Strings";

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
  let typeDescription = [{
    A string type containing a standard ASCII string.
  }];
}

def STRINGS_StringTensor : DialectType<
    STRINGS_Dialect,
    CPred<"$_self.isa<IREE::Strings::StringTensorType>()">,
    "string_tensor"> {
  let typeDescription = [{
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

  let arguments = (ins HAL_BufferView:$value);

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
  let summary = [{gathers all the strings from a Tensor by id}];
  let description = [{
    Gathers all the strings from a Tensor by ID.
  }];

  let arguments = (ins STRINGS_StringTensor:$dict,
                  HAL_BufferView:$ids);

  let results = (outs
    STRINGS_StringTensor:$result
  );
}

def STRINGS_PrintOp : Op<STRINGS_Dialect, "print"> {
  let summary = [{prints the contents of a string}];
  let description = [{
    Prints the contents of a string.
  }];

  let arguments = (ins STRINGS_String:$value);
}

#endif  // IREE_SAMPLES_STRINGS_MODULES_DIALECT_STRINGS_OPS
