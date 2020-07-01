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

#ifndef IREE_DIALECT_SEQUENCE_BASE
#define IREE_DIALECT_SEQUENCE_BASE

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

//===----------------------------------------------------------------------===//
// IREE sequence dialect
//===----------------------------------------------------------------------===//

// TODO(b/153499219): Flesh this description out more fully.

def SEQUENCE_Dialect : Dialect {
  let name = "sequence";
  let cppNamespace = "IREE::Sequence";

  let summary = [{
    A dialect designed to model iterable sequences.
  }];
  let description = [{
    The sequence dialect is used to model sequences that support high-level
    operations such as `map`, `sum`, and `reduce`.

    This dialect is currently under construction. Don't use it just yet.
  }];
}

//===----------------------------------------------------------------------===//
// Base sequence dialect op classes
//===----------------------------------------------------------------------===//

class SEQUENCE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SEQUENCE_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Sequence dialect types
//===----------------------------------------------------------------------===//

def SEQUENCE_AnySequenceType : DialectType<
    SEQUENCE_Dialect,
    CPred<"$_self.isa<Sequence::SequenceType>()">,
    "of"> {
  let typeDescription = [{
    Any type of sequence.
  }];
  let builderCall = "$_builder.getType<Sequence::SequenceType>()";
}

#endif  // IREE_DIALECT_SEQUENCE_BASE
