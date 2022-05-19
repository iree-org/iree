// RUN: iree-opt --split-input-file --verify-diagnostics --dispatch-transform-dialect-file-name=%p/dispatch_linalg_transform_spec.mlir | \
// RUN: --pass-pipeline="func.func(iree-flow-dispatch-linalg-on-tensors-transform-dialect-pass), cse, canonicalize, cse" %s 
//| FileCheck %s

func.func @tile_matmul_alone(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
             %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
