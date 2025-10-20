// RUN: iree-opt --iree-codegen-lower-tensor-ukernels --split-input-file --verify-diagnostics %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.ukernel_provider = #rocm.tensor_ukernel_provider}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  func.func private @ukernel_impl(tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.func @test(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = call @ukernel_impl(%arg0, %arg1, %arg2) : (tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
  func.func @replace_generic_with_ukernel_impl(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%1 : tensor<16x16xf32>) attrs =  {iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"test", tensor>} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
  }
}
// CHECK-LABEL: @ukernel_impl
// CHECK-LABEL: @replace_generic_with_ukernel_impl
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<16x32xf32>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<16x32xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         %[[OUT:.+]] = linalg.fill
// CHECK:         %[[CALL:.+]] = call @ukernel_impl(%[[LHS]], %[[RHS]], %[[OUT]]) : (tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:         return %[[CALL]]
