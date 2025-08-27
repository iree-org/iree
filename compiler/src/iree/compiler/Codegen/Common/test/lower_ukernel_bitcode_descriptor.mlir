// RUN: iree-opt --iree-codegen-lower-bitcode-ukernels --split-input-file %s | FileCheck %s

// CHECK-LABEL: @ukernel_test_without_provider
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
  func.func @ukernel_test_without_provider(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> tensor<16x16xf32> {
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

// -----

// CHECK-LABEL:       @pure_argmax_ukernel_test_with_provider
// CHECK-SAME:          %[[ARG0:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:          %[[ARG1:[a-zA-Z0-9]+]]: tensor<f32>
// CHECK-SAME:          %[[ARG2:[a-zA-Z0-9]+]]: tensor<i64>
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?xf32>
// CHECK:               %[[FALSE:.*]] = arith.constant false
// CHECK:               %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_amdgpu_argmax_f32i64"
// CHECK-SAME:            ins(%[[ARG0]] : tensor<?xf32>)
// CHECK-SAME:            outs(%[[ARG1]], %[[ARG2]] : tensor<f32>, tensor<i64>)
// CHECK-SAME:            (%[[DIM]], %[[FALSE]] : index, i1)
// CHECK-SAME:            fn_def_attrs {vm.import.module = "rocm"}
// CHECK-SAME{LITERAL}:   strided_dims([[], [], []])
// CHECK:               return %[[MICRO_KERNEL]]#1
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.ukernel_provider = #rocm.ukernel_provider}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  func.func @pure_argmax_ukernel_test_with_provider(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<i64>) -> tensor<i64> {
    %cst = arith.constant 0.000000e+00 : f32
    %0:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<?xf32>) outs(%arg1, %arg2 : tensor<f32>, tensor<i64>) attrs =  {iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"iree_uk_amdgpu_argmax_f32i64", bitcode>} {
    ^bb0(%in: f32, %out: f32, %out_0: i64):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i64
      %3 = arith.maximumf %in, %out : f32
      %4 = arith.cmpf ogt, %in, %out : f32
      %5 = arith.select %4, %2, %out_0 : i64
      linalg.yield %3, %5 : f32, i64
    } -> (tensor<f32>, tensor<i64>)
    return %0#1 : tensor<i64>
  }
}

// -----

// CHECK-LABEL:       @argmax_ukernel_test_with_provider
// CHECK-SAME:          %[[ARG0:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:          %[[ARG1:[a-zA-Z0-9]+]]: tensor<f32>
// CHECK-SAME:          %[[ARG2:[a-zA-Z0-9]+]]: tensor<i64>
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?xf32>
// CHECK:               %[[TRUE:.*]] = arith.constant true
// CHECK:               %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_amdgpu_argmax_f32i64"
// CHECK-SAME:            ins(%[[ARG0]] : tensor<?xf32>)
// CHECK-SAME:            outs(%[[ARG1]], %[[ARG2]] : tensor<f32>, tensor<i64>)
// CHECK-SAME:            (%[[DIM]], %[[TRUE]] : index, i1)
// CHECK-SAME:            fn_def_attrs {vm.import.module = "rocm"}
// CHECK-SAME{LITERAL}:   strided_dims([[], [], []])
// CHECK:               return %[[MICRO_KERNEL]]#0, %[[MICRO_KERNEL]]#1
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.ukernel_provider = #rocm.ukernel_provider}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  func.func @argmax_ukernel_test_with_provider(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<i64>) -> (tensor<f32>, tensor<i64>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<?xf32>) outs(%arg1, %arg2 : tensor<f32>, tensor<i64>) attrs =  {iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"iree_uk_amdgpu_argmax_f32i64", bitcode>} {
    ^bb0(%in: f32, %out: f32, %out_0: i64):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i64
      %3 = arith.maximumf %in, %out : f32
      %4 = arith.cmpf ogt, %in, %out : f32
      %5 = arith.select %4, %2, %out_0 : i64
      linalg.yield %3, %5 : f32, i64
    } -> (tensor<f32>, tensor<i64>)
    return %0#0, %0#1 : tensor<f32>, tensor<i64>
  }
}
