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

#ifndef IREE_SAMPLES_IREESTRINGS_MODULES_DIALECT_IREESTRINGS_OPS
#define IREE_SAMPLES_IREESTRINGS_MODULES_DIALECT_IREESTRINGS_OPS

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffects.td"

def IREESTRINGS_Dialect : Dialect {
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

def IREESTRINGS_String : DialectType<
    IREESTRINGS_Dialect,
    CPred<"$_self.isa<IREE::Strings::StringType>()">,
    "string"> {
  let typeDescription = [{
    A string type containing a standard ASCII string.
  }];
}

def IREESTRINGS_I32ToStringOp : Op<IREESTRINGS_Dialect, "i32_to_string", [NoSideEffect]> {
  let summary = [{converts an i32 to a string}];
  let description = [{
    Converts an i32 to its string representation.
  }];

  let arguments = (ins I32:$value);

  let results = (outs
    IREESTRINGS_String:$result
  );
}

def IREESTRINGS_PrintOp : Op<IREESTRINGS_Dialect, "print"> {
  let summary = [{prints the contents of a string}];
  let description = [{
    Prints the contents of a string.
  }];

  let arguments = (ins IREESTRINGS_String:$value);
}

#endif  // IREE_SAMPLES_IREESTRINGS_MODULES_DIALECT_IREESTRINGS_OPS
