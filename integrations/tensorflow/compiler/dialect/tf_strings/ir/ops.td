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
include "integrations/tensorflow/compiler/dialect/tf_strings/ir/base.td"

//===----------------------------------------------------------------------===//
// TFStrings dialect used for intermediate manipulations.
//===----------------------------------------------------------------------===//

def TFStrings_ToStringOp : TFStrings_Op<"to_string"> {
  let summary = "Converts the passed type to a string type.";
  let description = [{
    Prints the contents of a passed tensor.
  }];
  let arguments = (ins TFStrings_Any: $arguments);
  let results = (outs TFStrings_AnyString: $result);
  let builders = [OpBuilder<"Builder *, OperationState &state, Value value">];
}

def TFStrings_PrintOp : TFStrings_Op<"print"> {
  let summary = "Prints the string to standard output.";
  let description = [{
    Prints the contents of a passed tensor.
  }];
  let arguments = (ins TFStrings_AnyString:$arguments);
  let results = (outs );
}

#endif  // TFSTRINGS
