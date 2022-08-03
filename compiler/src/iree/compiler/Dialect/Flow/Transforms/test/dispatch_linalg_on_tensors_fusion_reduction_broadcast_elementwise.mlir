// RUN: iree-opt --iree-flow-fuse-reduction-broadcast-elementwise --split-input-file --pass-pipeline="func.func(iree-flow-dispatch-linalg-on-tensors-pass)" --canonicalize -cse %s | FileCheck %s
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @reduction_broadcast_elementwise_unary(%a: tensor<12x16x16xf32>, %b: tensor<12x16x16xf32>) -> tensor<12x16x16xf32> {
  %cst_47 = arith.constant 0.000000e+00 : f32
  %37 = linalg.init_tensor [12, 16] : tensor<12x16xf32>
  %38 = linalg.fill ins(%cst_47 : f32) outs(%37 : tensor<12x16xf32>) -> tensor<12x16xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a : tensor<12x16x16xf32>) outs(%38 : tensor<12x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
    %780 = arith.maxf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16xf32>
  %40 = linalg.init_tensor [12, 16, 16] : tensor<12x16x16xf32>
  %42 = linalg.generic {indexing_maps = [#map2, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%b, %39 : tensor<12x16x16xf32>, tensor<12x16xf32>) outs(%40 : tensor<12x16x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %780 = arith.subf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16x16xf32>
  return %42 : tensor<12x16x16xf32>
}

// There is only one input to the reduction.
// Check that two generic ops are dispatched together.
// The first generic (reduction) is directly used by the second generic (elementwise).

// CHECK-LABEL: func.func @reduction_broadcast_elementwise_unary
//      CHECK: flow.dispatch.workgroups
//      CHECK:   %[[RED:.+]] = linalg.generic
//      CHECK:   linalg.generic
//      CHECK-SAME: %[[RED]]

// -----

#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @reduction_broadcast_elementwise_binary1(%a1: tensor<128x384xf32>, %a2: tensor<128xf32>, %b: tensor<128x384xf32>) -> tensor<128x384xf32> {
  %cst_47 = arith.constant 0.000000e+00 : f32
  %37 = linalg.init_tensor [128] : tensor<128xf32>
  %38 = linalg.fill ins(%cst_47 : f32) outs(%37 : tensor<128xf32>) -> tensor<128xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%a1, %a2 : tensor<128x384xf32>, tensor<128xf32>) outs(%38 : tensor<128xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %585 = arith.subf %arg3, %arg4 : f32
      %586 = arith.mulf %585, %585 : f32
      %587 = arith.addf %586, %arg5 : f32
      linalg.yield %587 : f32
  } -> tensor<128xf32>
  %40 = linalg.init_tensor [128, 384] : tensor<128x384xf32>
  %42 = linalg.generic {indexing_maps = [#map2, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%b, %39 : tensor<128x384xf32>, tensor<128xf32>) outs(%40 : tensor<128x384xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %780 = arith.subf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<128x384xf32>
  return %42 : tensor<128x384xf32>
}

// There are two inputs to the reduction and one of them is broadcasted.
// Check that two generic ops are dispatched together.
// The first generic (reduction) is directly used by the second generic (elementwise).

// CHECK-LABEL: func.func @reduction_broadcast_elementwise_binary1
//      CHECK: flow.dispatch.workgroups
//      CHECK:   %[[RED:.+]] = linalg.generic
//      CHECK:   linalg.generic
//      CHECK-SAME: %[[RED]]

// -----

#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>

func.func @reduction_broadcast_elementwise_binary2(%a1: tensor<128x384xf32>, %a2: tensor<384xf32>, %b: tensor<128x384xf32>) -> tensor<128x384xf32> {
  %cst_47 = arith.constant 0.000000e+00 : f32
  %37 = linalg.init_tensor [128] : tensor<128xf32>
  %38 = linalg.fill ins(%cst_47 : f32) outs(%37 : tensor<128xf32>) -> tensor<128xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map3, #map1], iterator_types = ["parallel", "reduction"]} ins(%a1, %a2 : tensor<128x384xf32>, tensor<384xf32>) outs(%38 : tensor<128xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %585 = arith.subf %arg3, %arg4 : f32
      %586 = arith.mulf %585, %585 : f32
      %587 = arith.addf %586, %arg5 : f32
      linalg.yield %587 : f32
  } -> tensor<128xf32>
  %40 = linalg.init_tensor [128, 384] : tensor<128x384xf32>
  %42 = linalg.generic {indexing_maps = [#map2, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%b, %39 : tensor<128x384xf32>, tensor<128xf32>) outs(%40 : tensor<128x384xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %780 = arith.subf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<128x384xf32>
  return %42 : tensor<128x384xf32>
}

// There are two inputs to the reduction and one of them is broadcasted.
// Check that two generic ops are dispatched together.
// The first generic (reduction) is directly used by the second generic (elementwise).

// CHECK-LABEL: func.func @reduction_broadcast_elementwise_binary2
//      CHECK: flow.dispatch.workgroups
//      CHECK:   %[[RED:.+]] = linalg.generic
//      CHECK:   linalg.generic
//      CHECK-SAME: %[[RED]]

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @reduction_broadcast_elementwise_type_mismatch(%a: tensor<12x16x16xf32>, %b: tensor<12x16x32xf32>) -> tensor<12x16x32xf32> {
  %cst_47 = arith.constant 0.000000e+00 : f32
  %37 = linalg.init_tensor [12, 16] : tensor<12x16xf32>
  %38 = linalg.fill ins(%cst_47 : f32) outs(%37 : tensor<12x16xf32>) -> tensor<12x16xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a : tensor<12x16x16xf32>) outs(%38 : tensor<12x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
    %780 = arith.maxf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16xf32>
  %40 = linalg.init_tensor [12, 16, 32] : tensor<12x16x32xf32>
  %42 = linalg.generic {indexing_maps = [#map2, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%b, %39 : tensor<12x16x32xf32>, tensor<12x16xf32>) outs(%40 : tensor<12x16x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %780 = arith.subf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16x32xf32>
  return %42 : tensor<12x16x32xf32>
}

// Check that two generic ops are NOT dispatched together since the input type
// for reduction is different from the output type of the elementwise op. We
// should see two flow.dispatch.workgroups.

// CHECK-LABEL: func.func @reduction_broadcast_elementwise_type_mismatch
//      CHECK: flow.dispatch.workgroups
//      CHECK: flow.dispatch.workgroups


// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @reduction_broadcast_elementwise_dynamic(%a: tensor<12x16x?xf32>, %b: tensor<12x16x?xf32>) -> tensor<12x16x?xf32> {
  %cst_47 = arith.constant 0.000000e+00 : f32
  %37 = linalg.init_tensor [12, 16] : tensor<12x16xf32>
  %38 = linalg.fill ins(%cst_47 : f32) outs(%37 : tensor<12x16xf32>) -> tensor<12x16xf32>
  %39 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a : tensor<12x16x?xf32>) outs(%38 : tensor<12x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
    %780 = arith.maxf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16xf32>
  %c2 = arith.constant 2 : index
  %dim = tensor.dim %b, %c2 : tensor<12x16x?xf32>
  %40 = linalg.init_tensor [12, 16, %dim] : tensor<12x16x?xf32>
  %42 = linalg.generic {indexing_maps = [#map2, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%b, %39 : tensor<12x16x?xf32>, tensor<12x16xf32>) outs(%40 : tensor<12x16x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %780 = arith.subf %arg3, %arg4 : f32
    linalg.yield %780 : f32
  } -> tensor<12x16x?xf32>
  return %42 : tensor<12x16x?xf32>
}

// Dynamic shape case is not supported yet by the Vulkan codegen. See #9802.

// CHECK-LABEL: func.func @reduction_broadcast_elementwise_dynamic
//      CHECK: flow.dispatch.workgroups
//      CHECK: flow.dispatch.workgroups

