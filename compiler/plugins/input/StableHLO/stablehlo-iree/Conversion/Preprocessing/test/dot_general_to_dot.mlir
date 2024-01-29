// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-dot-general-to-dot))" \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: @testDebatch1
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x5x2xf32>, [[ARG1:%.+]]: tensor<2x3xf32>)
func.func @testDebatch1(%arg0: tensor<1x5x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x5x3xf32> {
  // CHECK-DAG: [[T0:%.+]] = stablehlo.transpose [[ARG0]], dims = [0, 1, 2]
  // CHECK-DAG: [[R0:%.+]] = stablehlo.reshape [[T0]] : (tensor<1x5x2xf32>) -> tensor<5x2xf32>
  // CHECK-DAG: [[T1:%.+]] = stablehlo.transpose [[ARG1]], dims = [0, 1]
  // CHECK:     [[R1:%.+]] = stablehlo.dot [[R0]], [[T1]], precision = [DEFAULT, DEFAULT] : (tensor<5x2xf32>, tensor<2x3xf32>) -> tensor<5x3xf32>
  // CHECK:     [[R2:%.+]] = stablehlo.reshape [[R1]] : (tensor<5x3xf32>) -> tensor<1x5x3xf32>
  // CHECK:     return [[R2]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >,
   precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x5x2xf32>, tensor<2x3xf32>) -> tensor<1x5x3xf32>

  func.return %0 : tensor<1x5x3xf32>
}

// -----

// CHECK-LABEL: @testDebatch2
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2x3xf32>, [[ARG1:%.+]]: tensor<1x5x2xf32>)
func.func @testDebatch2(%arg0: tensor<2x3xf32>, %arg1: tensor<1x5x2xf32>) -> tensor<3x1x5xf32> {
  // CHECK-DAG: [[R0:%.+]] = stablehlo.transpose [[ARG0]], dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK-DAG: [[R1:%.+]] = stablehlo.transpose [[ARG1]], dims = [2, 0, 1] : (tensor<1x5x2xf32>) -> tensor<2x1x5xf32>
  // CHECK-DAG: [[R2:%.+]] = stablehlo.reshape [[R1]] : (tensor<2x1x5xf32>) -> tensor<2x5xf32>
  // CHECK:     [[R3:%.+]] = stablehlo.dot [[R0]], [[R2]], precision = [DEFAULT, DEFAULT] : (tensor<3x2xf32>, tensor<2x5xf32>) -> tensor<3x5xf32>
  // CHECK:     [[R4:%.+]] = stablehlo.reshape [[R3]] : (tensor<3x5xf32>) -> tensor<3x1x5xf32>

  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x3xf32>, tensor<1x5x2xf32>) -> tensor<3x1x5xf32>
  func.return %0 : tensor<3x1x5xf32>
}

// -----

// CHECK-LABEL: @testBatchPassthrough
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2x2x3xf32>, [[ARG1:%.+]]: tensor<2x1x2xf32>)
func.func @testBatchPassthrough(%arg0: tensor<2x2x3xf32>, %arg1: tensor<2x1x2xf32>) -> tensor<2x3x1xf32> {
  // CHECK-NEXT:  stablehlo.dot_general [[ARG0]], [[ARG1]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x2x3xf32>, tensor<2x1x2xf32>) -> tensor<2x3x1xf32>
  func.return %0 : tensor<2x3x1xf32>
}

// -----

// CHECK-LABEL: @testVec
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<32xf32>, [[ARG1:%.+]]: tensor<32xf32>)
func.func @testVec(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<f32> {
  // CHECK-NEXT: [[R:%.+]] = stablehlo.dot [[ARG0]], [[ARG1]]
  // CHECK-NEXT: return [[R]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<32xf32>, tensor<32xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @testMatVec
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<20x32xf32>, [[ARG1:%.+]]: tensor<32xf32>)
func.func @testMatVec(%arg0: tensor<20x32xf32>, %arg1: tensor<32xf32>) -> tensor<20xf32> {
  // CHECK-NEXT: [[R:%.+]] = stablehlo.dot [[ARG0]], [[ARG1]]
  // CHECK-NEXT: return [[R]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<20x32xf32>, tensor<32xf32>) -> tensor<20xf32>
  func.return %0 : tensor<20xf32>
}

// -----

// CHECK-LABEL: @testMatVec
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<32x20xf32>, [[ARG1:%.+]]: tensor<32xf32>)
func.func @testMatVec(%arg0: tensor<32x20xf32>, %arg1: tensor<32xf32>) -> tensor<20xf32> {
  // CHECK-DAG:  [[T0:%.+]] = stablehlo.transpose [[ARG0]], dims = [1, 0]
  // CHECK-DAG:  [[T1:%.+]] = stablehlo.transpose [[ARG1]], dims = [0]
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.reshape [[T1]] : (tensor<32xf32>) -> tensor<32x1xf32>
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.reshape [[R1]] : (tensor<32x1xf32>) -> tensor<32xf32>
  // CHECK-NEXT: [[M:%.+]]  = stablehlo.dot [[T0]], [[R2]]
  // CHECK-NEXT: [[R1:%.+]]  = stablehlo.reshape [[M]]
  // CHECK-NEXT: [[R2:%.+]]  = stablehlo.reshape [[R1]]
  // CHECK-NEXT: return [[R2]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<32x20xf32>, tensor<32xf32>) -> tensor<20xf32>
  func.return %0 : tensor<20xf32>
}

// -----

// CHECK-LABEL: func @dot_general_to_dot_dynamic
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<128x4x?x32xf32>, [[ARG1:%.+]]: tensor<8x?x128x4xf32>)
func.func @dot_general_to_dot_dynamic(%arg0: tensor<128x4x?x32xf32>, %arg1: tensor<8x?x128x4xf32>) -> tensor<?x32x8x?xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [0, 1],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [2, 3],
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<128x4x?x32xf32>, tensor<8x?x128x4xf32>) -> tensor<?x32x8x?xf32>
  func.return %0 : tensor<?x32x8x?xf32>
}
// CHECK:     %[[C512:.+]] = stablehlo.constant dense<512> : tensor<1xi32>
// CHECK-DAG:                stablehlo.get_dimension_size [[ARG0]], dim = 2
// CHECK-DAG:                stablehlo.get_dimension_size [[ARG0]], dim = 3
// CHECK-DAG:                stablehlo.get_dimension_size [[ARG1]], dim = 0
// CHECK-DAG:                stablehlo.get_dimension_size [[ARG1]], dim = 0
// CHECK-DAG:                stablehlo.get_dimension_size [[ARG1]], dim = 1
// CHECK-DAG: %[[T0:.+]]   = stablehlo.transpose [[ARG0]], dims = [2, 3, 0, 1]
// CHECK-DAG: %[[R0:.+]]   = stablehlo.dynamic_reshape %[[T0]], {{%.+}} : (tensor<?x32x128x4xf32>, tensor<2xi32>) -> tensor<?x512xf32>
// CHECK-DAG: %[[T1:.+]]   = stablehlo.transpose [[ARG1]], dims = [2, 3, 0, 1]
// CHECK-DAG: %[[R1:.+]]   = stablehlo.dynamic_reshape %[[T1]], {{%.+}} : (tensor<128x4x8x?xf32>, tensor<2xi32>) -> tensor<512x?xf32>
// CHECK-DAG: %[[DOT:.+]]  = stablehlo.dot %[[R0]], %[[R1]], precision = [DEFAULT, DEFAULT] : (tensor<?x512xf32>, tensor<512x?xf32>) -> tensor<?x?xf32>
// CHECK:     %[[R2:.+]]   = stablehlo.dynamic_reshape %[[DOT]], {{%.+}} : (tensor<?x?xf32>, tensor<4xi32>) -> tensor<?x32x8x?xf32>
// CHECK-NEXT: return %[[R2]] : tensor<?x32x8x?xf32>

// -----

// CHECK-LABEL: func @dot_no_rhs_batch
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x512x768xf32>, [[ARG1:%.+]]: tensor<768x12x64xf32>)
func.func @dot_no_rhs_batch(%arg0: tensor<1x512x768xf32>, %arg1: tensor<768x12x64xf32>) -> tensor<1x512x12x64xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]>
    } : (tensor<1x512x768xf32>, tensor<768x12x64xf32>) -> tensor<1x512x12x64xf32>
  func.return %0 : tensor<1x512x12x64xf32>
}
// CHECK-DAG:  %[[T0:.+]]       = stablehlo.transpose [[ARG0]], dims = [0, 1, 2] : (tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
// CHECK-DAG:  %[[T1:.+]]       = stablehlo.transpose [[ARG1]], dims = [0, 1, 2] : (tensor<768x12x64xf32>) -> tensor<768x12x64xf32>
// CHECK-DAG:  %[[RESHAPEL:.+]] = stablehlo.reshape %[[T0]] : (tensor<1x512x768xf32>) -> tensor<512x768xf32>
// CHECK-DAG:  %[[RESHAPER:.+]] = stablehlo.reshape %[[T1]] : (tensor<768x12x64xf32>) -> tensor<768x768xf32>
// CHECK:      %[[DOT:.+]]      = stablehlo.dot %[[RESHAPEL]], %[[RESHAPER]] : (tensor<512x768xf32>, tensor<768x768xf32>) -> tensor<512x768xf32>
// CHECK:      %[[OUT:.+]]      = stablehlo.reshape %[[DOT]] : (tensor<512x768xf32>) -> tensor<1x512x12x64xf32>
// CHECK-NEXT: return %[[OUT]] : tensor<1x512x12x64xf32>

// -----

// CHECK-LABEL: @testPrefElem
func.func @testPrefElem(%arg0: tensor<1x1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x1x3xf64> {
  // CHECK: stablehlo.dot {{%.*}}, {{%.*}} precision = [DEFAULT, DEFAULT] : (tensor<2xf32>, tensor<2x3xf32>) -> tensor<3xf64>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >,
   precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x1x2xf32>, tensor<2x3xf32>) -> tensor<1x1x3xf64>

  func.return %0 : tensor<1x1x3xf64>
}

// -----

// CHECK-LABEL: @vecmat
func.func @vecmat(%arg0 : tensor<1x256xf32>, %arg1 : tensor<256x40xf32>) -> tensor<1x40xf32> {
  // CHECK: %[[R:.+]] = stablehlo.reshape %arg0 : (tensor<1x256xf32>) -> tensor<256xf32>
  // CHECK: %[[DOT:.+]] = stablehlo.dot %[[R]], %arg1, precision = [DEFAULT, DEFAULT] : (tensor<256xf32>, tensor<256x40xf32>) -> tensor<40xf32>
  // CHECK: %[[R:.+]] = stablehlo.reshape %[[DOT]] : (tensor<40xf32>) -> tensor<1x40xf32>
  %0 = "stablehlo.dot"(%arg0, %arg1) {precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x256xf32>, tensor<256x40xf32>) -> tensor<1x40xf32>

  // CHECK: return %[[R]]
  return %0 : tensor<1x40xf32>
}

// -----

// CHECK-LABEL: @matvec
func.func @matvec(%arg0 : tensor<20x144xf32>, %arg1 : tensor<1x144xf32>) -> tensor<20x1xf32> {
  // CHECK: %[[T0:.+]] = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<20x144xf32>) -> tensor<20x144xf32>
  // CHECK: %[[T1:.+]] = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<1x144xf32>) -> tensor<144x1xf32>
  // CHECK: %[[R0:.+]] = stablehlo.reshape %[[T1]] : (tensor<144x1xf32>) -> tensor<144xf32>
  // CHECK: %[[DOT:.+]] = stablehlo.dot %[[T0]], %[[R0]], precision = [DEFAULT, DEFAULT] : (tensor<20x144xf32>, tensor<144xf32>) -> tensor<20xf32>
  // CHECK: %[[R2:.+]] = stablehlo.reshape %[[DOT]] : (tensor<20xf32>) -> tensor<20x1xf32>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<20x144xf32>, tensor<1x144xf32>) -> tensor<20x1xf32>

  // CHECK: return %[[R2]]
  return %0 : tensor<20x1xf32>
}

// -----

// CHECK-LABEL: @vecdot
func.func @vecdot(%arg0 : tensor<1x32xf64>, %arg1 : tensor<32x1xf64>) -> tensor<1x1xf64> {
  // CHECK: %[[R0:.+]] = stablehlo.reshape %arg0 : (tensor<1x32xf64>) -> tensor<32xf64>
  // CHECK: %[[R1:.+]] = stablehlo.reshape %arg1 : (tensor<32x1xf64>) -> tensor<32xf64>
  // CHECK: %[[DOT:.+]] = stablehlo.dot %[[R0]], %[[R1]], precision = [DEFAULT, DEFAULT] : (tensor<32xf64>, tensor<32xf64>) -> tensor<f64>
  // CHECK: %[[R2:.+]] = stablehlo.reshape %[[DOT]] : (tensor<f64>) -> tensor<1x1xf64>
  %0 = "stablehlo.dot"(%arg0, %arg1) {precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x32xf64>, tensor<32x1xf64>) -> tensor<1x1xf64>

  // CHECK: %[[R2]]
  return %0 : tensor<1x1xf64>
}
