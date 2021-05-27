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

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_OPS_TD
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_OPS_TD

include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_base.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"

class TFTensorList_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TFTensorList_Dialect, mnemonic, traits> {
}

def TFTensorList_Reserve : TFTensorList_Op<"Reserve"> {
  let summary = "Return a TensorList of the given size with empty elements.";

  let description = [{
    Returns a TensorList with capacity for `num_elements` of presumed partial
    shape `element_shape`.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$element_shape,
    I32Tensor:$num_elements,
    TypeAttr:$element_type
  );

  let results = (outs
    TFTensorList_TensorList:$list
  );
}

def TFTensorList_GetItem : TFTensorList_Op<"GetItem"> {
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
    TFTensorList_TensorList:$list,
    I32Tensor:$index
  );

  let results = (outs
    TF_Tensor:$item
  );
}

def TFTensorList_SetItem : TFTensorList_Op<"SetItem"> {
  let summary = "Sets an item of the TensorList, returning an updated list.";

  let description = [{
    Returns a new TensorList which is the same as `list` but with the item at
    index `index` replaced with `item`.
  }];

  let arguments = (ins
    TFTensorList_TensorList:$list,
    I32Tensor:$index,
    TF_Tensor:$item
  );

  let results = (outs
    TFTensorList_TensorList:$new_list
  );
}

def TFTensorList_FromTensor : TFTensorList_Op<"FromTensor", [NoSideEffect]> {
  let summary = [{
    Creates a TensorList which, when stacked, has the value of `tensor`.
  }];

  let description = [{
    Each tensor in the result list corresponds to one row of the input tensor.

    tensor: The input tensor.
    list: The list.
  }];

  let arguments = (ins
    TF_Tensor:$tensor
  );

  let results = (outs
    TFTensorList_TensorList:$list
  );
}

def TFTensorList_Stack : TFTensorList_Op<"Stack", [NoSideEffect]> {
  let summary = "Stacks all tensors in the list.";

  let description = [{
    Requires that all tensors have the same shape.

    list: the input list
    tensor: the gathered result
    num_elements: optional. If not -1, the number of elements in the list.
  }];

  let arguments = (ins
    TFTensorList_TensorList:$list,
    // TODO(silvasean): Properly handle IREE's blind truncation to 32-bit.
    // This is logically `index` type, but coming from TensorFlow it
    // comes in as i64. IREE then proceeds to blindly truncate it to I32
    // in-place. So relax this from I64Tensor to I32Or64Tensor.
    TF_I32OrI64Tensor:$num_elements
  );

  let results = (outs
    TF_Tensor:$tensor
  );
}

def TFTensorList_Concat : TFTensorList_Op<"Concat", [NoSideEffect]> {
  let summary = "Concatenates all tensors in the list along the first dimension.";

  let description = [{
    Requires that all tensors have the same shape along dimensions 1+.

    list: the input list
    tensor: the concatenated result
  }];

  let arguments = (ins
    TFTensorList_TensorList:$list
  );

  let results = (outs
    TF_Tensor:$tensor
  );
}

// TODO(suderman): If gradients are not being calculated, the value will be
// DCE'd.
def TFTensorList_GetDim0 : TFTensorList_Op<"GetDim0", [NoSideEffect]> {
  let summary = "Returns a tensor of the lengths of Dim0 for the tensorlists' tensors";

  let description = [{
    Requires that all tensors have 1+ dimensions.

    list: the input list
    tensor: a 1-D tensor of length size of Dim-1 for each tensor.
  }];

  let arguments = (ins
    TFTensorList_TensorList:$list
  );

  let results = (outs
    TF_Tensor:$tensor
  );
}

#endif // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_DIALECT_TF_TENSORLIST_IR_TF_TENSORLIST_OPS_TD
