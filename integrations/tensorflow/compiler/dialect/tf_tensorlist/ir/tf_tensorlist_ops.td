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

#ifndef IREE_INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_OPS_TD
#define IREE_INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_OPS_TD

include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_base.td"
include "mlir/IR/OpBase.td"
include "mlir/Analysis/InferTypeOpInterface.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"

class TfTensorList_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TfTensorList_Dialect, mnemonic, traits> {
}

def TfTensorList_ToVariant : TfTensorList_Op<"ToVariant", [NoSideEffect]> {
  let summary = "Convert tensorflow variant to tensorlist";
  let description = [{
    This op is created transiently as part of the conversion process from
    tf dialect to tf_tensorlist and bridges between the two type systems.
    // TODO(silvasean): Avoid needing this.
  }];
  let arguments = (ins
    TfTensorList_TensorList:$list
  );
  let results = (outs
    TF_VariantTensor:$variant
  );
}

// TODO(silvasean): Is there a cleaner way to factor InferTypeOpInterface?
// Right now we need the verbose
// `DeclareOpInterfaceMethods<InferTypeOpInterface>` and then
// the implementation of FromVariant::inferReturnTypes in the .cc file.
// It would be nice if we could just say `ReturnsTensorList` in the trait list.

def TfTensorList_FromVariant :
    TfTensorList_Op<"FromVariant",
      [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Convert tensorlist to tensorflow variant";
  let description = [{
    This op is created transiently as part of the conversion process from
    tf dialect to tf_tensorlist and bridges between the two type systems.
    // TODO(silvasean): Avoid needing this.
  }];
  let arguments = (ins
    TF_VariantTensor:$variant
  );
  let results = (outs
    TfTensorList_TensorList:$list
  );
  let hasCanonicalizer = 1;
}

def TfTensorList_Reserve :
    TfTensorList_Op<"Reserve",
      [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Return a TensorList of the given size with empty elements.";

  let description = [{
    Returns a TensorList with capacity for `num_elements` of presumed partial
    shape `element_shape`.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$element_shape,
    I32Tensor:$num_elements
  );

  let results = (outs
    TfTensorList_TensorList:$list
  );
}

def TfTensorList_GetItem : TfTensorList_Op<"GetItem"> {
  let summary = "Returns an item of the TensorList.";

  let description = [{
    Returns the item at index `index` from the TensorList.

    If index is out of bounds of the valid tensors, but is in bounds of the
    reserved capacity, then `element_shape` controls the shape of the returned
    value, which will be filled with zeros.

    If element_shape is a fully defined shape (that is, contains no `-1` values),
    then zeros of that shape are returned. Otherwise, that shape is merged with
    the shape of all the other valid tensors in the list (that is an O(N)
    operation), and the resulting shape is used for creating the zeros.

    See the TensorFlow [implementation](https://github.com/tensorflow/tensorflow/blob/2e6a3c58e4b96cac864f244e4886ef00b3184986/tensorflow/core/kernels/list_kernels.h#L168).
  }];

  let arguments = (ins
    TfTensorList_TensorList:$list,
    I32Tensor:$index,
    I32Tensor:$element_shape
  );

  let results = (outs
    TF_Tensor:$item
  );
}

def TfTensorList_SetItem :
    TfTensorList_Op<"SetItem",
      [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Sets an item of the TensorList, returning an updated list.";

  let description = [{
    Returns a new TensorList which is the same as `list` but with the item at
    index `index` replaced with `item`.
  }];

  let arguments = (ins
    TfTensorList_TensorList:$list,
    I32Tensor:$index,
    TF_Tensor:$item
  );

  let results = (outs
    TfTensorList_TensorList:$new_list
  );
}
#endif // IREE_INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_OPS_TD
