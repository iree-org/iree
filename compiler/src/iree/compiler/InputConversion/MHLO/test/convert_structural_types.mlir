// RUN: iree-opt --split-input-file --iree-mhlo-to-linalg-on-tensors %s | FileCheck %s

// CHECK-LABEL: @func_cfg_conversion
module @func_cfg_conversion {
  // CHECK: func.func @caller(%arg0: tensor<2xui32>, %arg1: i1) -> tensor<2xui32>
  func.func @caller(%arg0: tensor<2xui32>, %arg1 : i1) -> tensor<2xui32> {
    // CHECK: %[[RESULT:.*]] = call @callee(%arg0, %arg1) : (tensor<2xui32>, i1) -> tensor<2xui32>
    %1 = call @callee(%arg0, %arg1) : (tensor<2xui32>, i1) -> tensor<2xui32>
    // CHECK: return %[[RESULT]] : tensor<2xui32>
    return %1 : tensor<2xui32>
  }

  // CHECK: func.func @callee(%arg0: tensor<2xui32>, %arg1: i1) -> tensor<2xui32>
  func.func @callee(%arg0: tensor<2xui32>, %arg1: i1) -> tensor<2xui32> {
    // CHECK: cf.cond_br %arg1, ^bb1(%arg0 : tensor<2xui32>), ^bb2(%arg0 : tensor<2xui32>)
    cf.cond_br %arg1, ^bb1(%arg0 : tensor<2xui32>), ^bb2(%arg0 : tensor<2xui32>)
  // CHECK: ^bb1(%[[BB1_PHI:.*]]: tensor<2xui32>)
  ^bb1(%phi0 : tensor<2xui32>) :
    // CHECK: tensor.bitcast %[[BB1_PHI]] : tensor<2xui32> to tensor<2xi32> 
    // CHECK: %[[BB1_PHI_ADD:.*]] = linalg.generic
    // CHECK: %[[BB1_PHI_ADD_CAST:.*]] = tensor.bitcast %[[BB1_PHI_ADD]] : tensor<2xi32> to tensor<2xui32> 
    // CHECK: cf.br ^bb2(%[[BB1_PHI_ADD_CAST]] : tensor<2xui32>)
    %0 = "mhlo.add"(%phi0, %phi0) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    cf.br ^bb2(%0 : tensor<2xui32>)
  // CHECK: ^bb2(%[[BB2_PHI:.*]]: tensor<2xui32>)
  ^bb2(%phi1 : tensor<2xui32>):
    // CHECK: tensor.bitcast %[[BB2_PHI]] : tensor<2xui32> to tensor<2xi32> 
    // CHECK: %[[BB2_PHI_ADD:.*]] = linalg.generic
    // CHECK: %[[BB2_PHI_ADD_CAST:.*]] = tensor.bitcast %[[BB2_PHI_ADD]] : tensor<2xi32> to tensor<2xui32> 
    // CHECK: return %[[BB2_PHI_ADD_CAST]] : tensor<2xui32>
    %1 = "mhlo.add"(%phi1, %phi1) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    return %1 : tensor<2xui32>
  }
}
