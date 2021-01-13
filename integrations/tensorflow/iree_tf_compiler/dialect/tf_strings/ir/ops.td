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

// Common IREE op definitions shared by the interpreter and sequencer dialects.

#ifndef TFSTRINGS
#define TFSTRINGS

include "mlir/IR/OpBase.td"
include "iree_tf_compiler/dialect/tf_strings/ir/base.td"

//===----------------------------------------------------------------------===//
// TFStrings dialect used for intermediate manipulations.
//===----------------------------------------------------------------------===//

def TFStrings_PrintOp : TFStrings_Op<"print"> {
  let summary = "Prints the string to standard output.";
  let description = [{
    Prints the contents of a passed tensor.
  }];
  let arguments = (ins TFStrings_String:$arguments);
  let results = (outs);
}

def TFStrings_ToStringOp : TFStrings_Op<"to_string"> {
  let summary = "Converts the value to a string.";
  let description = [{
    Converts the value to a string.
  }];
  let arguments = (ins TFStrings_Value: $arguments);
  let results = (outs TFStrings_String: $result);
  let builders = [
    OpBuilderDAG<(ins "Value":$value)>];
}

def TFStrings_GatherOp : Op<TFStrings_Dialect, "gather"> {
  let summary = "Gathers all the strings from by index.";
  let description = [{
    Gathers all the strings from a Tensor using the index of the last dimension.
  }];

  let arguments = (ins TFStrings_StringTensor:$dict,
                  TFStrings_ValueTensor:$indices);

  let results = (outs
    TFStrings_StringTensor:$result
  );
}

def TFStrings_ConcatOp : Op<TFStrings_Dialect, "concat"> {
  let summary = "Concatenates the strings in the tensor along the last dimension.";

  let description = [{
    Concatenates the strings in the tensor along the last dimension.
  }];

  let arguments = (ins TFStrings_StringTensor:$value);

  let results = (outs
    TFStrings_StringTensor:$result
  );
}

def TFStrings_ToStringTensorOp : TFStrings_Op<"to_string_tensor"> {
  let summary = "Converts a tensor of values to a tensor of strings.";
  let description = [{
    Converts a tensor of values to a tensor of strings.
  }];
  let arguments = (ins TFStrings_ValueTensor: $arguments);
  let results = (outs TFStrings_StringTensor: $result);
  let builders = [
    OpBuilderDAG<(ins "Value":$value)>];
}

def TFStrings_StringTensorToStringOp : TFStrings_Op<"string_tensor_to_string"> {
  let summary = "Converts a string tensor to a string.";
  let description = [{
    Converts a tensor of strings to a string.
  }];
  let arguments = (ins TFStrings_StringTensor: $arguments);
  let results = (outs TFStrings_String: $result);
  let builders = [
    OpBuilderDAG<(ins "Value":$value)>];
}

#endif  // TFSTRINGS
