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

#ifndef IREE_DIALECT_SEQUENCE_OPS
#define IREE_DIALECT_SEQUENCE_OPS

include "iree/compiler/Dialect/Sequence/IR/SequenceBase.td"

def Sequence_MapOp : SEQUENCE_Op<"map", []> {
  let summary = [{applies a map function to a sequence element-wise}];
  let description = [{
    Returns a sequence resulting from applying a mapping function elementwise.
  }];

  // TODO(b/153499219): Add a way for the mapping function to capture additional
  // input from the outer context, or supply an additional input parameter to
  // enable things like what's expressed in this pseudocode:
  // x = ...
  // y = sequence.map(z -> z + x, ...)
  let arguments = (ins
    FlatSymbolRefAttr:$mapping_function,
    SEQUENCE_AnySequenceType:$input_sequence
  );
  let results = (outs
    SEQUENCE_AnySequenceType:$output_sequence
  );
  let assemblyFormat = "$mapping_function `,` $input_sequence  `:` type($input_sequence) `->` type($output_sequence) attr-dict";
  let verifier = [{ return verifyMapOp(*this); }];
}

// TODO(b/153499219): Add ways to construct new sequences.
// TODO(b/153499219): Add `sequence.reduce` and `sequence.sum` ops.
// TODO(b/153499219): Add lowering to something executable.
// TODO(b/153499219): Add whatever is needed for parity with `tf.data`.

#endif  // IREE_DIALECT_SEQUENCE_OPS
