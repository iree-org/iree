// Copyright 2019 Google LLC
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

#ifndef IREE_DIALECT_MODULES_TENSORLIST_BASE
#define IREE_DIALECT_MODULES_TENSORLIST_BASE

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

def TensorList_Dialect : Dialect {
  let name = "tensorlist";
  let cppNamespace = "IREE::TensorList";

  let summary = [{
    A dialect for implementing TensorFlow's TensorList data structure.
  }];
  let description = [{
    This dialect contains ops that map 1:1 to the corresponding IREE native
    module.

    This dialect mimics TensorFlow's TensorList op set. It provides a
    a straightforward implementation of those semantics.
  }];
}

def TensorList_TensorList : DialectType<
    TensorList_Dialect,
    CPred<"$_self.isa<IREE::TensorList::TensorListType>()">,
    "tensorlist.list"> {
  let typeDescription = [{
    IREE VM type representing a TensorList.
  }];
}


#endif  // IREE_DIALECT_MODULES_TENSORLIST_BASE
