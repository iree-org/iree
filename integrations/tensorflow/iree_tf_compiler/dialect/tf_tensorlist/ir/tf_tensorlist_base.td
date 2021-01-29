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

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_BASE_TD
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_BASE_TD

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

def TFTensorList_Dialect : Dialect {
  let name = "tf_tensorlist";

  let summary = "Dialect used to represent TensorFlow TensorList's in MLIR.";

  let description = [{
    This is a "frontend" dialect intended to map 1:1 to TensorFlow's TensorList
    ops. The main difference from the `tf` dialect is that this dialect
    has a dedicated TensorList type, instead of using a `tensor<!tf.variant>`
    to represent TensorLists.

    This allows representing the TensorList construct with greater fidelity.
  }];

  let cppNamespace = "::mlir::iree_integrations::tf_tensorlist";
}

def TFTensorList_TensorList :
    Type<CPred<"$_self.isa<mlir::iree_integrations::tf_tensorlist::TensorListType>()">, "list">;

#endif  // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_BASE_TD
