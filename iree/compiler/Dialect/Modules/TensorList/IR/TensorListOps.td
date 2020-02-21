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

#ifndef IREE_DIALECT_MODULES_TENSORLIST_OPS
#define IREE_DIALECT_MODULES_TENSORLIST_OPS

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"
include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListBase.td"
include "mlir/IR/OpBase.td"

def TensorList_Reserve : Op<TensorList_Dialect, "Reserve"> {
  let summary = [{Create a new tensorlist with a given capacity.}];
  let description = [{
    Create a new tensorlist with a given capacity.
  }];

  let arguments = (ins
    // TODO(silvasean): Do we need element_shape?
    HAL_BufferView:$element_shape,
    // TODO(silvasean): Convert to `I32:$count` instead.
    HAL_BufferView:$count
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
    HAL_BufferView:$index,
    // TODO(silvasean): Do we need element_shape?
    HAL_BufferView:$element_shape
  );

  let results = (outs
    HAL_BufferView:$item
  );
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
    HAL_BufferView:$index,
    HAL_BufferView:$item
  );

  let results = (outs
    TensorList_TensorList:$new_list
  );
}

#endif  // IREE_DIALECT_MODULES_TENSORLIST_OPS
