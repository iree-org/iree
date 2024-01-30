// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-quantized-conv-to-conv))" -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @conv2d_zp
func.func @conv2d_zps(%arg0: tensor<1x14x16x5xi8>, %arg1: tensor<3x4x5x1024xi8>, %arg2: tensor<1x12x13x1024xi32>) -> tensor<1x12x13x1024xi32> {
  %iZp = arith.constant 0 : i32
  %fZp = arith.constant 0 : i32
  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
  // CHECK-SAME:  dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:  strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:  ins(%arg0, %arg1 : tensor<1x14x16x5xi8>, tensor<3x4x5x1024xi8>)
  // CHECK-SAME:  outs(%arg2 : tensor<1x12x13x1024xi32>)
  %0 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, %iZp, %fZp : tensor<1x14x16x5xi8>, tensor<3x4x5x1024xi8>, i32, i32) outs(%arg2 : tensor<1x12x13x1024xi32>) -> tensor<1x12x13x1024xi32>

  // CHECK: return %[[CONV]]
  return %0 : tensor<1x12x13x1024xi32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3) -> (d3)>

// CHECK-LABEL: func.func @conv2d_filter_zp
func.func @conv2d_filter_zp(%arg0: tensor<1x14x16x5xi8>, %arg1: tensor<3x4x5x1024xi8>, %arg2: tensor<1x12x13x1024xi32>) -> tensor<1x12x13x1024xi32> {
  %iZp = arith.constant 42 : i32
  %fZp = arith.constant 0 : i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0
  // CHECK-DAG: %[[C42:.+]] = arith.constant 42
  // CHECK-DAG: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
  // CHECK-DAG: %[[SUM_INIT:.+]] = tensor.empty() : tensor<1024xi32>
  // CHECK-DAG: %[[SUM_FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%1 : tensor<1024xi32>) -> tensor<1024xi32>

  // CHECK: %[[SUM:.+]] = linalg.generic
  // CHECK-SAME:  indexing_maps = [#map, #map1]
  // CHECK-SAME:  iterator_types = ["parallel", "reduction", "reduction", "reduction"]}
  // CHECK-SAME:  ins(%arg1 : tensor<3x4x5x1024xi8>)
  // CHECK-SAME:  outs(%[[SUM_FILL]] : tensor<1024xi32>)
  // CHECK:   ^bb0(%[[IN:.+]]: i8, %[[OUT:.+]]: i32):
  // CHECK:   %[[EXT:.+]] = arith.extsi %[[IN]] : i8 to i32
  // CHECK:   %[[ADD:.+]] = arith.addi %[[EXT]], %[[OUT]] : i32
  // CHECK:   linalg.yield %[[ADD]]

  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x12x13x1024xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME:  indexing_maps = [#map2, #map3, #map2]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME:  ins(%[[CONV]], %[[SUM]] : tensor<1x12x13x1024xi32>, tensor<1024xi32>)
  // CHECK-SAME:  outs(%[[INIT]] : tensor<1x12x13x1024xi32>)
  // CHECK:   ^bb0(%[[A0:.+]]: i32, %[[A1:.+]]: i32, %[[A2:.+]]: i32):
  // CHECK:   %[[MUL:.+]] = arith.muli %[[A1]], %[[C42]] : i32
  // CHECK:   %[[SUB:.+]] = arith.subi %[[A0]], %[[MUL]] : i32
  // CHECK:   linalg.yield %[[SUB]]
  %0 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, %iZp, %fZp : tensor<1x14x16x5xi8>, tensor<3x4x5x1024xi8>, i32, i32) outs(%arg2 : tensor<1x12x13x1024xi32>) -> tensor<1x12x13x1024xi32>

  // CHECK: return %[[GENERIC]] : tensor<1x12x13x1024xi32>
  return %0 : tensor<1x12x13x1024xi32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @conv2d_input_zp
func.func @conv2d_input_zp(%arg0: tensor<1x14x16x5xi8>, %arg1: tensor<3x4x5x1024xi8>, %arg2: tensor<1x12x13x1024xi32>) -> tensor<1x12x13x1024xi32> {
  %iZp = arith.constant 0 : i32
  %fZp = arith.constant 42 : i32

  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C42:.+]] = arith.constant 42 : i32
  // CHECK-DAG: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf

  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x14x16xi32>
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[INIT]] : tensor<1x14x16xi32>)
  // CHECK: %[[SUM:.+]] = linalg.generic
  // CHECK-SAME:  indexing_maps = [#map, #map1]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME:  ins(%arg0 : tensor<1x14x16x5xi8>)
  // CHECK-SAME:  outs(%[[FILL]] : tensor<1x14x16xi32>)

  // CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[SUM]] {{\[\[}}0], [1], [2, 3]] : tensor<1x14x16xi32> into tensor<1x14x16x1xi32>
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[INIT]] : tensor<1x12x13x1xi32>)
  // CHECK: %[[KERNEL:.+]] = tensor.empty() : tensor<3x4xi32>
  // CHECK: %[[POOL:.+]] = linalg.pooling_nhwc_sum
  // CHECK-SAME:  dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:  strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:  ins(%[[EXPAND]], %[[KERNEL]] : tensor<1x14x16x1xi32>, tensor<3x4xi32>)
  // CHECK-SAME:  outs(%[[FILL]] : tensor<1x12x13x1xi32>)
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[POOL]] {{\[\[}}0], [1], [2, 3]]

  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x12x13x1024xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME:  indexing_maps = [#map, #map1, #map]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME:  ins(%[[CONV]], %[[COLLAPSE]] : tensor<1x12x13x1024xi32>, tensor<1x12x13xi32>)
  // CHECK-SAME:  outs(%[[INIT]] : tensor<1x12x13x1024xi32>)
  // CHECK:   ^bb0(%[[A0:.+]]: i32, %[[A1:.+]]: i32, %[[A2:.+]]: i32):
  // CHECK:   %[[MUL:.+]] = arith.muli %[[A1]], %[[C42]]
  // CHECK:   %[[SUB:.+]] = arith.subi %[[A0]], %[[MUL]]
  // CHECK:   linalg.yield %[[SUB]]
  %0 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, %iZp, %fZp : tensor<1x14x16x5xi8>, tensor<3x4x5x1024xi8>, i32, i32) outs(%arg2 : tensor<1x12x13x1024xi32>) -> tensor<1x12x13x1024xi32>

  // CHECK: return %[[GENERIC]]
  return %0 : tensor<1x12x13x1024xi32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @conv2d_full(%arg0: tensor<1x14x16x5xi8>, %arg1: tensor<3x4x5x1024xi8>, %arg2: tensor<1x12x13x1024xi32>) -> tensor<1x12x13x1024xi32> {
  %iZp = arith.constant 17 : i32
  %fZp = arith.constant 42 : i32

  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C42840:.+]] = arith.constant 42840 : i32
  // CHECK-DAG: %[[C17:.+]] = arith.constant 17 : i32
  // CHECK-DAG: %[[C42:.+]] = arith.constant 42 : i32

  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf

  // CHECK: %[[SUM:.+]] = linalg.generic
  // CHECK-SAME:  ins(%arg1 : tensor<3x4x5x1024xi8>)
  // CHECK: %[[IZP:.+]] = linalg.generic
  // CHECK-SAME:  ins(%[[CONV]], %[[SUM]] : tensor<1x12x13x1024xi32>, tensor<1024xi32>)

  // CHECK: %[[SUM:.+]] = linalg.generic
  // CHECK-SAME:  ins(%arg0 : tensor<1x14x16x5xi8>)
  // CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[SUM]]
  // CHECK: %[[KERNEL:.+]] = tensor.empty() : tensor<3x4xi32>
  // CHECK: %[[POOL:.+]] = linalg.pooling_nhwc_sum
  // CHECK-SAME:  ins(%[[EXPAND]], %[[KERNEL]] : tensor<1x14x16x1xi32>, tensor<3x4xi32>)
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[POOL]]
  // CHECK: %[[FZP:.+]] = linalg.generic
  // CHECK-SAME:  ins(%[[IZP]], %[[COLLAPSE]] : tensor<1x12x13x1024xi32>, tensor<1x12x13xi32>)

  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x12x13x1024xi32>
  // CHECK: %[[FINAL:.+]] = linalg.generic {indexing_maps = [#map2, #map2]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME:  ins(%[[FZP]] : tensor<1x12x13x1024xi32>)
  // CHECK-SAME:  outs(%[[INIT]] : tensor<1x12x13x1024xi32>)
  // CHECK:   ^bb0(%[[A0:.+]]: i32, %[[A1:.+]]: i32):
  // CHECK:   %[[ADD:.+]] = arith.addi %[[A0]], %[[C42840]] : i32
  // CHECK:   linalg.yield %[[ADD]]
  %0 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, %iZp, %fZp : tensor<1x14x16x5xi8>, tensor<3x4x5x1024xi8>, i32, i32) outs(%arg2 : tensor<1x12x13x1024xi32>) -> tensor<1x12x13x1024xi32>

  // CHECK: return %[[FINAL]]
  return %0 : tensor<1x12x13x1024xi32>
}

// -----

// CHECK-LABEL: func.func @conv2d_dyn
func.func @conv2d_dyn(%arg0: tensor<?x?x?x5xi8>, %arg1: tensor<3x4x5x1024xi8>, %arg2: tensor<?x?x?x1024xi32>) -> tensor<?x?x?x1024xi32> {
  %iZp = arith.constant 0 : i32
  %fZp = arith.constant 0 : i32
  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
  %0 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, %iZp, %fZp : tensor<?x?x?x5xi8>, tensor<3x4x5x1024xi8>, i32, i32) outs(%arg2 : tensor<?x?x?x1024xi32>) -> tensor<?x?x?x1024xi32>

  // CHECK: return %[[CONV]]
  return %0 : tensor<?x?x?x1024xi32>
}

// -----

// CHECK-LABEL: @conv2d_dyn_filter_zp
func.func @conv2d_dyn_filter_zp(%arg0: tensor<1x14x16x5xi8>, %arg1: tensor<?x?x5x1024xi8>, %arg2: tensor<1x?x?x1024xi32>) -> tensor<1x?x?x1024xi32> {
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index

  %iZp = arith.constant 42 : i32
  %fZp = arith.constant 0 : i32

  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf
  // CHECK: %[[SUM:.+]] = linalg.generic
  // CHECK-SAME: ins(%arg1 : tensor<?x?x5x1024xi8>)

  // CHECK: %[[DIM1:.+]] = tensor.dim %arg2, %c1 : tensor<1x?x?x1024xi32>
  // CHECK: %[[DIM2:.+]] = tensor.dim %arg2, %c2 : tensor<1x?x?x1024xi32>
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DIM1]], %[[DIM2]]) : tensor<1x?x?x1024xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME: ins(%[[CONV]], %[[SUM]] : tensor<1x?x?x1024xi32>, tensor<1024xi32>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<1x?x?x1024xi32>)
  %0 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, %iZp, %fZp : tensor<1x14x16x5xi8>, tensor<?x?x5x1024xi8>, i32, i32) outs(%arg2 : tensor<1x?x?x1024xi32>) -> tensor<1x?x?x1024xi32>

  // CHECK: return %[[GENERIC]]
  return %0 : tensor<1x?x?x1024xi32>
}

// -----

// CHECK-LABEL: @conv2d_dyn_input_zp
func.func @conv2d_dyn_input_zp(%arg0: tensor<1x14x16x5xi8>, %arg1: tensor<?x?x5x1024xi8>, %arg2: tensor<1x?x?x1024xi32>) -> tensor<1x?x?x1024xi32> {
  %fZp = arith.constant 42 : i32
  %iZp = arith.constant 0 : i32

  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[I1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[I2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[I0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C42:.+]] = arith.constant 42 : i32
  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf

  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x14x16xi32>
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[INIT]] : tensor<1x14x16xi32>)
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME: ins(%arg0 : tensor<1x14x16x5xi8>) outs(%[[FILL]] : tensor<1x14x16xi32>)
  // CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[GENERIC]]

  // CHECK: %[[DIM1:.+]] = tensor.dim %arg2, %[[I1]]
  // CHECK: %[[DIM2:.+]] = tensor.dim %arg2, %[[I2]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DIM1]], %[[DIM2]]) : tensor<1x?x?x1xi32>
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[INIT]] : tensor<1x?x?x1xi32>)
  // CHECK: %[[DIM0:.+]] = tensor.dim %arg1, %[[I0]]
  // CHECK: %[[DIM1:.+]] = tensor.dim %arg1, %[[I1]]
  // CHECK: %[[KERNEL:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xi32>
  // CHECK: %[[POOL:.+]] = linalg.pooling_nhwc_sum
  // CHECK-SAME: ins(%[[EXPAND]], %[[KERNEL]] : tensor<1x14x16x1xi32>, tensor<?x?xi32>)
  // CHECK-SAME: outs(%[[FILL]] : tensor<1x?x?x1xi32>)
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[POOL]]

  // CHECK: %[[DIM1:.+]] = tensor.dim %arg2, %[[I1]]
  // CHECK: %[[DIM2:.+]] = tensor.dim %arg2, %[[I2]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DIM1]], %[[DIM2]]) : tensor<1x?x?x1024xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK-SAME: ins(%[[CONV]], %[[COLLAPSE]] : tensor<1x?x?x1024xi32>, tensor<1x?x?xi32>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<1x?x?x1024xi32>)
  %0 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, %iZp, %fZp : tensor<1x14x16x5xi8>, tensor<?x?x5x1024xi8>, i32, i32) outs(%arg2 : tensor<1x?x?x1024xi32>) -> tensor<1x?x?x1024xi32>

  // CHECK: return %[[GENERIC]]
  return %0 : tensor<1x?x?x1024xi32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: @conv2d_all_dyn
func.func @conv2d_all_dyn(%arg0: tensor<?x?x?x?xi8>, %arg1: tensor<?x?x?x?xi8>, %arg2: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %fZp = arith.constant 42 : i32
  %iZp = arith.constant 13 : i32

  // CHECK-DAG: %[[I3:.+]] = arith.constant 3 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[I0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[I1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[I2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[C546:.+]] = arith.constant 546 : i32
  // CHECK-DAG: %[[C42:.+]] = arith.constant 42 : i32
  // CHECK-DAG: %[[C13:.+]] = arith.constant 13 : i32
  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-SAME:  ins(%arg0, %arg1 : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>)
  // CHECK-SAME:  outs(%arg2 : tensor<?x?x?x?xi32>)

  // CHECK: %[[DIM3:.+]] = tensor.dim %arg1, %[[I3]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%[[DIM3]]) : tensor<?xi32>
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<?xi32>)
  // CHECK: %[[FSUM:.+]] = linalg.generic
  // CHECK-SAME:  indexing_maps = [#map, #map1]
  // CHECK-SAME:  iterator_types = ["parallel", "reduction", "reduction", "reduction"]
  // CHECK-SAME:  ins(%arg1 : tensor<?x?x?x?xi8>)
  // CHECK-SAME:  outs(%[[FILL]] : tensor<?xi32>)

  // CHECK: %[[DIM0:.+]] = tensor.dim %arg0, %[[I0]]
  // CHECK: %[[DIM1:.+]] = tensor.dim %arg2, %[[I1]]
  // CHECK: %[[DIM2:.+]] = tensor.dim %arg2, %[[I2]]
  // CHECK: %[[DIM3:.+]] = tensor.dim %arg1, %[[I3]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]])
  // CHECK: %[[CONV_SUMF:.+]] = linalg.generic
  // CHECK-SAME:  indexing_maps = [#map2, #map3, #map2]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK-SAME:  ins(%[[CONV]], %[[FSUM]] : tensor<?x?x?x?xi32>, tensor<?xi32>)
  // CHECK-SAME:  outs(%[[EMPTY]] : tensor<?x?x?x?xi32>)

  // CHECK: %[[DIM0:.+]] = tensor.dim %arg0, %[[I0]]
  // CHECK: %[[DIM1:.+]] = tensor.dim %arg0, %[[I1]]
  // CHECK: %[[DIM2:.+]] = tensor.dim %arg0, %[[I2]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]])
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<?x?x?xi32>)
  // CHECK: %[[SUMI:.+]] = linalg.generic
  // CHECK-SAME:  indexing_maps = [#map2, #map4]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
  // CHECK-SAME:  ins(%arg0 : tensor<?x?x?x?xi8>)
  // CHECK-SAME:  outs(%[[FILL]] : tensor<?x?x?xi32>)
  // CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[SUMI]] {{\[\[}}0], [1], [2, 3]]

  // CHECK-DAG: %[[DIM0:.+]] = tensor.dim %arg0, %[[I0]]
  // CHECK-DAG: %[[DIM1:.+]] = tensor.dim %arg2, %[[I1]]
  // CHECK-DAG: %[[DIM2:.+]] = tensor.dim %arg2, %[[I2]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]])
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%9 : tensor<?x?x?x1xi32>)
  // CHECK-DAG: %[[DIM0:.+]] = tensor.dim %arg1, %[[I0]] : tensor<?x?x?x?xi8>
  // CHECK-DAG: %[[DIM1:.+]] = tensor.dim %arg1, %[[I1]] : tensor<?x?x?x?xi8>
  // CHECK: %[[KERNEL:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xi32>
  // CHECK: %[[POOL:.+]] = linalg.pooling_nhwc_sum
  // CHECK-SAME: dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME: strides = dense<1> : tensor<2xi64>}
  // CHECK-SAME: ins(%[[EXPAND]], %[[KERNEL]] : tensor<?x?x?x1xi32>, tensor<?x?xi32>)
  // CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?x1xi32>)
  // CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[POOL]] {{\[\[}}0], [1], [2, 3]]

  // CHECK-DAG: %[[DIM0:.+]] = tensor.dim %arg0, %[[I0]]
  // CHECK-DAG: %[[DIM1:.+]] = tensor.dim %arg2, %[[I1]]
  // CHECK-DAG: %[[DIM2:.+]] = tensor.dim %arg2, %[[I2]]
  // CHECK-DAG: %[[DIM3:.+]] = tensor.dim %arg1, %[[I3]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]])
  // CHECK: %[[CONV_SUMIF:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map2, #map4, #map2]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[CONV_SUMF]], %[[COLLAPSE]] : tensor<?x?x?x?xi32>, tensor<?x?x?xi32>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<?x?x?x?xi32>)

  // CHECK: %[[DIM0:.+]] = tensor.dim %arg1, %[[I0]]
  // CHECK-DAG: %[[DIM1:.+]] = tensor.dim %arg1, %[[I1]]
  // CHECK-DAG: %[[DIM2:.+]] = tensor.dim %arg1, %[[I2]]
  // CHECK-DAG: %[[MUL1:.+]] = arith.muli %[[DIM0]], %[[DIM1]]
  // CHECK-DAG: %[[MUL2:.+]] = arith.muli %[[MUL1]], %[[DIM2]]
  // CHECK-DAG: %[[CAST:.+]] = arith.index_cast %[[MUL2]] : index to i32
  // CHECK-DAG: %[[DIM0:.+]] = tensor.dim %arg0, %[[I0]]
  // CHECK-DAG: %[[DIM1:.+]] = tensor.dim %arg2, %[[I1]]
  // CHECK-DAG: %[[DIM2:.+]] = tensor.dim %arg2, %[[I2]]
  // CHECK-DAG: %[[DIM3:.+]] = tensor.dim %arg1, %[[I3]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]]) : tensor<?x?x?x?xi32>
  // CHECK: %[[RESULT:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map2, #map2]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[CONV_SUMIF]] : tensor<?x?x?x?xi32>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<?x?x?x?xi32>)
  %0 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, %iZp, %fZp : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>, i32, i32) outs(%arg2 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>

  // CHECK: return %[[RESULT]]
  return %0 : tensor<?x?x?x?xi32>
}

// -----

// CHECK-LABEL: func.func @dconv2d
func.func @dconv2d(%arg0 : tensor<1x64x64x16xi8>, %arg1 : tensor<4x4x16xi8>, %arg2 : tensor<1x61x61x16xi32>) -> tensor<1x61x61x16xi32> {
  %iZp = arith.constant 0 : i32
  %fZp = arith.constant 0 : i32

  // CHECK: %[[RET:.+]] = linalg.depthwise_conv_2d_nhwc_hwc {
  // CHECK-SAME:    dilations = dense<1> : tensor<2xi64>,
  // CHECK-SAME:    strides = dense<1> : tensor<2xi64>}
  // CHECK-SAME:    ins(%arg0, %arg1 : tensor<1x64x64x16xi8>, tensor<4x4x16xi8>)
  // CHECK-SAME:    outs(%arg2 : tensor<1x61x61x16xi32>) -> tensor<1x61x61x16xi32>
  %result = linalg.depthwise_conv_2d_nhwc_hwc_q {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    }
    ins(%arg0, %arg1, %iZp, %fZp : tensor<1x64x64x16xi8>, tensor<4x4x16xi8>, i32, i32)
    outs(%arg2 : tensor<1x61x61x16xi32>) -> tensor<1x61x61x16xi32>
  // CHECK: %[[RET]]
  return %result : tensor<1x61x61x16xi32>
}

// -----

// CHECK-LABEL: func.func @dconv2d_izp
func.func @dconv2d_izp(%arg0 : tensor<1x64x64x16xi8>, %arg1 : tensor<4x4x16xi8>, %arg2 : tensor<1x61x61x16xi32>) -> tensor<1x61x61x16xi32> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i32
  %iZp = arith.constant 2 : i32
  %fZp = arith.constant 0 : i32

  // CHECK: %[[CONV:.+]] = linalg.depthwise_conv_2d_nhwc_hwc

  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<16xi32>
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<16xi32>)
  // CHECK: %[[SUM:.+]] = linalg.generic
  // CHECK-SAME:      indexing_maps = [#map, #map1]
  // CHECK-SAME:      iterator_types = ["parallel", "reduction", "reduction"]
  // CHECK-SAME:      ins(%arg1 : tensor<4x4x16xi8>)
  // CHECK-SAME:      outs(%[[FILL]] : tensor<16xi32>) {
  // CHECK:   ^bb0(%[[IN:.+]]: i8, %[[OUT:.+]]: i32):
  // CHECK:    %[[EXT:.+]] = arith.extsi %[[IN]] : i8 to i32
  // CHECK:    %[[ADD:.+]] = arith.addi %6, %[[OUT]]
  // CHECK:    linalg.yield %[[ADD]]

  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x61x61x16xi32>
  // CHECK: %[[UPDATE:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map2, #map3, #map2]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[CONV]], %[[SUM]] : tensor<1x61x61x16xi32>, tensor<16xi32>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<1x61x61x16xi32>)
  // CHECK: ^bb0(%[[IN0:.+]]: i32, %[[IN1:.+]]: i32, %[[OUT:.+]]: i32):
  // CHECK:   %[[MUL:.+]] = arith.muli %[[IN1]], %[[C2]] : i32
  // CHECK:   %[[SUB:.+]] = arith.subi %[[IN0]], %[[MUL]] : i32
  // CHECK:   linalg.yield %[[SUB]] : i32

  %result = linalg.depthwise_conv_2d_nhwc_hwc_q {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    }
    ins(%arg0, %arg1, %iZp, %fZp : tensor<1x64x64x16xi8>, tensor<4x4x16xi8>, i32, i32)
    outs(%arg2 : tensor<1x61x61x16xi32>) -> tensor<1x61x61x16xi32>
  // CHECK: return %[[UPDATE]]
  return %result : tensor<1x61x61x16xi32>
}

// -----

// CHECK-LABEL: @dconv2d_fzp
func.func @dconv2d_fzp(%arg0 : tensor<1x64x64x16xi8>, %arg1 : tensor<4x4x16xi8>, %arg2 : tensor<1x61x61x16xi32>) -> tensor<1x61x61x16xi32> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i32
  %iZp = arith.constant 0 : i32
  %fZp = arith.constant 2 : i32

  // CHECK: %[[CONV:.+]] = linalg.depthwise_conv_2d_nhwc_hwc

  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x61x61x16xi32>
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<1x61x61x16xi32>)
  // CHECK: %[[WINDOW:.+]] = tensor.empty() : tensor<4x4xi32>
  // CHECK: %[[POOL:.+]] = linalg.pooling_nhwc_sum
  // CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:      strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:      ins(%arg0, %[[WINDOW]] : tensor<1x64x64x16xi8>, tensor<4x4xi32>)
  // CHECK-SAME:      outs(%[[FILL]] : tensor<1x61x61x16xi32>)

  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x61x61x16xi32>
  // CHECK:  %[[UPDATE:.+]] = linalg.generic
  // CHECK-SAME:      indexing_maps = [#map, #map, #map]
  // CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME:      ins(%[[CONV]], %[[POOL]] : tensor<1x61x61x16xi32>, tensor<1x61x61x16xi32>)
  // CHECK-SAME:      outs(%[[EMPTY]] : tensor<1x61x61x16xi32>)
  // CHECK:  ^bb0(%[[IN0:.+]]: i32, %[[IN1:.+]]: i32, %[[OUT:.+]]: i32):
  // CHECK:    %[[MUL:.+]] = arith.muli %[[IN1]], %[[C2]] : i32
  // CHECK:    %[[SUB:.+]] = arith.subi %[[IN0]], %[[MUL]] : i32
  // CHECK:    linalg.yield %[[SUB]]
  %result = linalg.depthwise_conv_2d_nhwc_hwc_q {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    }
    ins(%arg0, %arg1, %iZp, %fZp : tensor<1x64x64x16xi8>, tensor<4x4x16xi8>, i32, i32)
    outs(%arg2 : tensor<1x61x61x16xi32>) -> tensor<1x61x61x16xi32>

  // CHECK: return %[[UPDATE]]
  return %result : tensor<1x61x61x16xi32>
}

// -----

// CHECK-LABEL: func.func @dconv2d_ifzp
func.func @dconv2d_ifzp(%arg0 : tensor<1x64x64x16xi8>, %arg1 : tensor<4x4x16xi8>, %arg2 : tensor<1x61x61x16xi32>) -> tensor<1x61x61x16xi32> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1232:.+]] = arith.constant 1232 : i32
  // CHECK-DAG: %[[C7:.+]] = arith.constant 7 : i32
  // CHECK-DAG: %[[C11:.+]] = arith.constant 11 : i32
  %iZp = arith.constant 7 : i32
  %fZp = arith.constant 11 : i32

  // CHECK: %[[DCONV:.+]] = linalg.depthwise_conv_2d_nhwc_hwc

  // CHECK: %[[SUMF:.+]] = linalg.generic
  // CHECK-SAME: ins(%arg1 : tensor<4x4x16xi8>)
  // CHECK: %[[FIX_IZP:.+]] = linalg.generic
  // CHECK-SAME: ins(%[[DCONV]], %[[SUMF]] : tensor<1x61x61x16xi32>, tensor<16xi32>)

  // CHECK: %[[WINDOW:.+]] = tensor.empty() : tensor<4x4xi32>
  // CHECK: %[[POOL:.+]] = linalg.pooling_nhwc_sum
  // CHECK-SAME: ins(%arg0, %[[WINDOW]] : tensor<1x64x64x16xi8>, tensor<4x4xi32>)
  // CHECK: %[[FIX_FZP:.+]] = linalg.generic
  // CHECK:  ins(%[[FIX_IZP]], %[[POOL]] : tensor<1x61x61x16xi32>, tensor<1x61x61x16xi32>)

  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x61x61x16xi32>
  // CHECK: %[[RET:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#map2, #map2]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[FIX_FZP]] : tensor<1x61x61x16xi32>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<1x61x61x16xi32>)
  // CHECK: ^bb0(%[[IN:.+]]: i32, %[[OUT:.+]]: i32):
  // CHECK:   %[[ADD:.+]] = arith.addi %[[IN]], %[[C1232]] : i32
  // CHECK:   linalg.yield %[[ADD]]
  %result = linalg.depthwise_conv_2d_nhwc_hwc_q {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    }
    ins(%arg0, %arg1, %iZp, %fZp : tensor<1x64x64x16xi8>, tensor<4x4x16xi8>, i32, i32)
    outs(%arg2 : tensor<1x61x61x16xi32>) -> tensor<1x61x61x16xi32>

  // CHECK: return %[[RET]]
  return %result : tensor<1x61x61x16xi32>
}

// -----

// CHECK-LABEL: @dconv2d_dyn
func.func @dconv2d_dyn(%arg0 : tensor<?x?x?x?xi8>, %arg1 : tensor<?x?x?xi8>, %arg2 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  // CHECK-DAG: %[[I0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[I1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[I2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[I3:.+]] = arith.constant 3 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C7:.+]] = arith.constant 7 : i32
  // CHECK-DAG: %[[C11:.+]] = arith.constant 11 : i32
  // CHECK-DAG: %[[C77:.+]] = arith.constant 77 : i32

  %iZp = arith.constant 7 : i32
  %fZp = arith.constant 11 : i32

  // CHECK: %[[DCONV:.+]] = linalg.depthwise_conv_2d_nhwc_hwc

  // CHECK: %[[DIM:.+]] = tensor.dim %arg1, %[[I2]] : tensor<?x?x?xi8>
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%[[DIM]]) : tensor<?xi32>
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<?xi32>) -> tensor<?xi32>
  // CHECK: %[[SUMF:.+]] = linalg.generic
  // CHECK-SAME: ins(%arg1 : tensor<?x?x?xi8>)
  // CHECK-SAME: outs(%[[FILL]] : tensor<?xi32>)

  // CHECK: %[[DIM0:.+]] = tensor.dim %arg0, %[[I0]]
  // CHECK: %[[DIM1:.+]] = tensor.dim %arg2, %[[I1]]
  // CHECK: %[[DIM2:.+]] = tensor.dim %arg2, %[[I2]]
  // CHECK: %[[DIM3:.+]] = tensor.dim %arg0, %[[I3]]
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]]) : tensor<?x?x?x?xi32>
  // CHECK: %[[FIX_IZP:.+]] = linalg.generic
  // CHECK-SAME: ins(%[[DCONV]], %[[SUMF]] : tensor<?x?x?x?xi32>, tensor<?xi32>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<?x?x?x?xi32>)

  // CHECK:  %[[DIM0:.+]] = tensor.dim %arg0, %[[I0]]
  // CHECK:  %[[DIM1:.+]] = tensor.dim %arg2, %[[I1]]
  // CHECK:  %[[DIM2:.+]] = tensor.dim %arg2, %[[I2]]
  // CHECK:  %[[DIM3:.+]] = tensor.dim %arg0, %[[I3]]
  // CHECK:  %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]]) : tensor<?x?x?x?xi32>
  // CHECK:  %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<?x?x?x?xi32>)

  // CHECK: %[[DIM0:.+]] = tensor.dim %arg1, %[[I0]] : tensor<?x?x?xi8>
  // CHECK: %[[DIM1:.+]] = tensor.dim %arg1, %[[I1]] : tensor<?x?x?xi8>
  // CHECK: %[[WINDOW:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xi32>
  // CHECK: %[[POOL:.+]] = linalg.pooling_nhwc_sum
  // CHECK-SAME: ins(%arg0, %[[WINDOW]] : tensor<?x?x?x?xi8>, tensor<?x?xi32>)
  // CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?x?xi32>)

  // CHECK:  %[[DIM0:.+]] = tensor.dim %arg0, %[[I0]]
  // CHECK:  %[[DIM1:.+]] = tensor.dim %arg2, %[[I1]]
  // CHECK:  %[[DIM2:.+]] = tensor.dim %arg2, %[[I2]]
  // CHECK:  %[[DIM3:.+]] = tensor.dim %arg0, %[[I3]]
  // CHECK:  %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]]) : tensor<?x?x?x?xi32>
  // CHECK: %[[FIX_FZP:.+]] = linalg.generic
  // CHECK-SAME: ins(%[[FIX_IZP]], %[[POOL]] : tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<?x?x?x?xi32>)

  // CHECK: %[[DIM0:.+]] = tensor.dim %arg1, %[[I0]]
  // CHECK: %[[DIM1:.+]] = tensor.dim %arg1, %[[I1]]
  // CHECK: %[[MUL:.+]] = arith.muli %[[DIM0]], %[[DIM1]] : index
  // CHECK: %[[CAST:.+]] = arith.index_cast %[[MUL]] : index to i32
  // CHECK: %[[IFZP:.+]] = arith.muli %[[CAST]], %[[C77]]

  // CHECK:  %[[DIM0:.+]] = tensor.dim %arg0, %[[I0]]
  // CHECK:  %[[DIM1:.+]] = tensor.dim %arg2, %[[I1]]
  // CHECK:  %[[DIM2:.+]] = tensor.dim %arg2, %[[I2]]
  // CHECK:  %[[DIM3:.+]] = tensor.dim %arg0, %[[I3]]
  // CHECK:  %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]]) : tensor<?x?x?x?xi32>
  // CHECK: %[[RET:.+]] = linalg.generic
  // CHECK-SAME: ins(%[[FIX_FZP]] : tensor<?x?x?x?xi32>)
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<?x?x?x?xi32>)
  // CHECK:    ^bb0(%[[IN:.+]]: i32, %{{.+}}: i32):
  // CHECK:    %[[ADD:.+]] = arith.addi %[[IN]], %[[IFZP]]
  // CHECK:    linalg.yield %[[ADD]]
  %result = linalg.depthwise_conv_2d_nhwc_hwc_q {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    }
    ins(%arg0, %arg1, %iZp, %fZp : tensor<?x?x?x?xi8>, tensor<?x?x?xi8>, i32, i32)
    outs(%arg2 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>

  // CHECK: return %[[RET]]
  return %result : tensor<?x?x?x?xi32>
}

// -----

// Make sure linalg.depthwise_conv_2d_nhwd_hwcm_q is converted to
// linalg.depthwise_conv_2d_nhwc_hwc_q and then lowered according to the
// patterns tested above.

// CHECK-LABEL: @dconv2d_hwcm
//   CHECK-NOT:   linalg.depthwise_conv_2d_nhwc_hwcm_q
//       CHECK:   linalg.depthwise_conv_2d_nhwc_hwc
func.func @dconv2d_hwcm(%arg0: tensor<?x?x?x?xi8>, %arg1: tensor<?x?x?x1xi8>, %arg2: tensor<?x?x?x?x1xi32>, %arg3 : i32, %arg4 : i32) -> tensor<?x?x?x?x1xi32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm_q {_someattr, dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1, %arg3, %arg4 : tensor<?x?x?x?xi8>, tensor<?x?x?x1xi8>, i32, i32) outs(%arg2 : tensor<?x?x?x?x1xi32>) -> tensor<?x?x?x?x1xi32>
  return %0 : tensor<?x?x?x?x1xi32>
}
