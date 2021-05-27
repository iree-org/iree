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

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TF_TENSORLIST_OPS_H_
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TF_TENSORLIST_OPS_H_

#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_types.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

// Declares the operations for this dialect using the generated header.
#define GET_OP_CLASSES
#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.h.inc"

#endif  // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TF_TENSORLIST_OPS_H_
