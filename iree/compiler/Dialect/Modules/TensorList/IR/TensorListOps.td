// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_TENSORLIST_OPS
#define IREE_DIALECT_MODULES_TENSORLIST_OPS

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"
include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListBase.td"
include "mlir/IR/OpBase.td"

def TensorList_TensorOrBuffer : AnyTypeOf<[HAL_BufferView, AnyRankedTensor]>;

def TensorList_ReserveTensor : Op<TensorList_Dialect, "Reserve.Tensor"> {
  let summary = [{Create a new tensorlist with a given capacity.}];
  let description = [{
    Create a new tensorlist with a given capacity.
  }];

  let arguments = (ins
    AnyRankedTensor:$element_shape,
    AnyRankedTensor:$count,
    TypeAttr:$element_type
  );

  let results = (outs
    TensorList_TensorList:$list
  );
}

def TensorList_Reserve : Op<TensorList_Dialect, "Reserve"> {
  let summary = [{Create a new tensorlist with a given capacity.}];
  let description = [{
    Create a new tensorlist with a given capacity.
  }];

  let arguments = (ins
    HAL_BufferView:$element_shape,
    // TODO(silvasean): Convert to `I32:$count` instead.
    HAL_BufferView:$count,
    HAL_ElementTypeAttr:$element_type
  );

  let results = (outs
    TensorList_TensorList:$list
  );
}

def TensorList_GetItem : Op<TensorList_Dialect, "GetItem"> {
  let summary = [{Gets an item out of a tensorlist.}];
  let description = [{
    Gets an item out of a tensorlist.
  }];

  let arguments = (ins
    TensorList_TensorList:$list,
    // TODO(silvasean): Convert to `I32:$index` instead.
    TensorList_TensorOrBuffer:$index
  );

  let results = (outs TensorList_TensorOrBuffer:$item);
}

def TensorList_SetItem : Op<TensorList_Dialect, "SetItem"> {
  let summary = [{Sets an item of a tensorlist.}];
  let description = [{
    Sets an item of a tensorlist, returning a new tensorlist `new_list`
    reflecting the updated value. Does not mutate `list`.
  }];

  let arguments = (ins
    TensorList_TensorList:$list,
    // TODO(silvasean): Convert to `I32:$index` instead.
    TensorList_TensorOrBuffer:$index,
    TensorList_TensorOrBuffer:$item
  );

  let results = (outs
    TensorList_TensorList:$new_list
  );
}

def TensorList_FromTensor : Op<TensorList_Dialect, "FromTensor"> {
  let summary = "Creates a tensorlist by slicing along the first dimension";
  let description = [{
    The tensor `tensor` will be sliced along its first dimension, resulting in
    a tensorlist `list` of length equal to `tensor`'s leading dimension.
  }];
  let arguments = (ins
    TensorList_TensorOrBuffer:$tensor
  );
  let results = (outs
    TensorList_TensorList:$list
  );
}

def TensorList_StackTensor : Op<TensorList_Dialect, "Stack.Tensor"> {
  let summary = "Creates a tensor by stacking the tensors in the tensorlist";
  let description = [{
    Creates a new tensor `tensor` by stacking the tensors in `list`.
    The resulting tensor has a leading dimension equal to the length of `list`.

    Requires the list to be non-empty.
    Requires all tensors contained in `list` to be the same shape.
  }];
  let arguments = (ins
    TensorList_TensorList:$list,
    AnyRankedTensor:$num_elements
  );
  let results = (outs
    AnyRankedTensor:$tensor
  );
}

def TensorList_Stack : Op<TensorList_Dialect, "Stack"> {
  let summary = "Creates a tensor by stacking the tensors in the tensorlist";
  let description = [{
    Creates a new tensor `tensor` by stacking the tensors in `list`.
    The resulting tensor has a leading dimension equal to the length of `list`.

    Requires the list to be non-empty.
    Requires all tensors contained in `list` to be the same shape.
  }];
  let arguments = (ins
    HAL_Allocator:$allocator,
    TensorList_TensorList:$list,
    HAL_BufferView:$num_elements
  );
  let results = (outs
    HAL_BufferView:$tensor
  );
}

def TensorList_ConcatTensor : Op<TensorList_Dialect, "Concat.Tensor"> {
  let summary = "Creates a tensor by concatenate the tensors in the tensorlist";
  let description = [{
    Creates a new tensor `tensor` by concatenating the tensors in `list` along
    the leading dimension.

    Requires the list to be non-empty.
    Requires all tensors contained in `list` to have the same dimensions along
    the non-leading axes.
  }];
  let arguments = (ins
    TensorList_TensorList:$list
  );
  let results = (outs
    AnyRankedTensor:$tensor
  );
}

def TensorList_Concat : Op<TensorList_Dialect, "Concat"> {
  let summary = "Creates a tensor by concatenate the tensors in the tensorlist";
  let description = [{
    Creates a new tensor `tensor` by concatenating the tensors in `list` along
    the leading dimension.

    Requires the list to be non-empty.
    Requires all tensors contained in `list` to have the same dimensions along
    the non-leading axes.
  }];
  let arguments = (ins
    HAL_Allocator:$allocator,
    TensorList_TensorList:$list
  );
  let results = (outs
    HAL_BufferView:$tensor
  );
}

#endif  // IREE_DIALECT_MODULES_TENSORLIST_OPS
