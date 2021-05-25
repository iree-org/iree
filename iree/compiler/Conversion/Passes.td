
// Copyright 2021 Google LLC
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

#ifndef IREE_CONVERSION_PASSES
#define IREE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertHLOToLinalgOnTensors :
    Pass<"iree-codegen-hlo-to-linalg-on-tensors", "FuncOp"> {
  let summary = "Convert from XLA-HLO ops to Linalg ops on tensors";
  let constructor = "mlir::iree_compiler::createHLOToLinalgOnTensorsPass()";
}

def DemoteF32ToF16 :
    Pass<"iree-convert-f32-to-f16", "ModuleOp"> {
  let summary = "Convert f32 operations and values into equivalent f16 ones.";
  let constructor = "mlir::iree_compiler::createDemoteF32ToF16Pass()";
}

def FusionOfTensorOps :
    Pass<"iree-codegen-fusion-of-tensor-ops", ""> {
  let summary = "Fuse operations on tensors";
  let constructor = "mlir::iree_compiler::createFusionOfTensorOpsPass()";
}

def ResolveShapeOps :
    Pass<"iree-codegen-resolve-shape", "FuncOp"> {
  let summary = "resolve shapes";
  let constructor = "mlir::iree_compiler::createResolveShapeOpsPass()";
}

#endif  // IREE_DIALECT_FLOW_PASSES
