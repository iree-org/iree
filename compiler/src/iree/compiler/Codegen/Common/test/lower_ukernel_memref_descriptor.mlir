// RUN: iree-opt --iree-codegen-lower-memref-ukernels %s | FileCheck %s

// CHECK-LABEL: @ukernel_impl
// CHECK-LABEL: @replace_generic_with_ukernel_impl
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: memref<16x32xf32>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: memref<16x32xf32>
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]: memref<16x16xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         linalg.fill
// CHECK:         call @ukernel_impl(%[[LHS]], %[[RHS]], %[[OUT]]) : (memref<16x32xf32>, memref<16x32xf32>, memref<16x16xf32>) -> ()
// CHECK:         return
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.ukernel_provider = #iree_codegen.symbolic_ukernel_provider}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  func.func private @ukernel_impl(memref<16x32xf32>, memref<16x32xf32>, memref<16x16xf32>)
  func.func @test(%arg0: memref<16x32xf32>, %arg1: memref<16x32xf32>, %arg2: memref<16x16xf32>) {
    call @ukernel_impl(%arg0, %arg1, %arg2) : (memref<16x32xf32>, memref<16x32xf32>, memref<16x16xf32>) -> ()
    return
  }
  func.func @replace_generic_with_ukernel_impl(%arg0: memref<16x32xf32>, %arg1: memref<16x32xf32>, %arg2: memref<16x16xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%arg2 : memref<16x16xf32>)
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<16x32xf32>, memref<16x32xf32>) outs(%arg2 : memref<16x16xf32>) attrs =  {iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"test", memref>} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    return
  }
}
