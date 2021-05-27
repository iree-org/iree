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
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.td"

// GetElementTypeAttr
def GetVariantElementTypeAttr : NativeCodeCall<
  "GetVariantElementTypeAttr($0.getType())">;

def : Pat<(TF_TensorListReserveOp:$result $element_shape, $num_elements),
          (TFTensorList_Reserve $element_shape, $num_elements,
           (GetVariantElementTypeAttr $result))>;

def : Pat<(TF_TensorListGetItemOp $input_handle, $index, $element_shape),
          (TFTensorList_GetItem $input_handle, $index)>;

def : Pat<(TF_TensorListSetItemOp $input_handle, $index, $item),
          (TFTensorList_SetItem $input_handle, $index, $item)>;

def : Pat<(TF_TensorListFromTensorOp $tensor, $element_shape),
          (TFTensorList_FromTensor $tensor)>;

def WrapScalarI64InTensor : NativeCodeCall<
    "DenseElementsAttr::get(RankedTensorType::get({}, $_builder.getIntegerType(64)), {$_self.getValue()})">;
def : Pat<(TF_TensorListStackOp $input_handle, $element_shape, $num_elements),
          (TFTensorList_Stack
            $input_handle,
            (ConstantOp WrapScalarI64InTensor:$num_elements))>;
