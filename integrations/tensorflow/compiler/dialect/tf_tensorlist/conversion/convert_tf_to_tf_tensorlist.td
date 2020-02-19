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

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.td"

def : Pat<(TF_TensorListReserveOp $element_shape, $num_elements),
          (TfTensorList_ToVariant
            (TfTensorList_Reserve $element_shape, $num_elements))>;

def : Pat<(TF_TensorListGetItemOp $input_handle, $index, $element_shape),
          (TfTensorList_GetItem
            (TfTensorList_FromVariant $input_handle),
            $index,
            $element_shape)>;

def : Pat<(TF_TensorListSetItemOp $input_handle, $index, $item),
          (TfTensorList_ToVariant
            (TfTensorList_SetItem
              (TfTensorList_FromVariant $input_handle),
              $index,
              $item))>;

