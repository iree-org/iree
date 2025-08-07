// RUN: iree-opt --iree-codegen-lower-bitcode-ukernels %s | FileCheck %s

// CHECK-LABEL: @ukernel_test
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<16x32xf32>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<16x32xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         %[[OUT:.+]] = linalg.fill
// CHECK:         %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "test"
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] : tensor<16x32xf32>, tensor<16x32xf32>)
// CHECK-SAME:       outs(%[[OUT]] : tensor<16x16xf32>)
// CHECK:         return %[[MICRO_KERNEL]]
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  func.func @ukernel_test(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%1 : tensor<16x16xf32>) attrs =  {iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"test", bitcode>} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
  }
}
