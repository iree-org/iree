// RUN: iree-opt %s --iree-stablehlo-to-linalg --split-input-file \
// RUN:   --canonicalize | FileCheck %s

// CHECK-LABEL: func @rng_uniform_1d
func.func @rng_uniform_1d(%min: tensor<f32>, %max: tensor<f32>) -> tensor<10xf32> {
  %shape = arith.constant dense<[10]>  : tensor<1xi32>
  %0 = "stablehlo.rng"(%min, %max, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<1xi32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
// CHECK-DAG:  ^{{.+}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[IDX0_CAST]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL2_CAST:.+]] = arith.uitofp %[[VAL2]] : i32 to f32
// CHECK-DAG:    %[[VAL4:.+]] = arith.mulf %[[VAL2_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL5:.+]] = arith.addf %[[VAL4]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL5]] : f32
// CHECK-NEXT: -> tensor<10xf32>

// -----

// CHECK-LABEL: func @rng_uniform_2d
func.func @rng_uniform_2d(%min: tensor<f32>, %max: tensor<f32>) -> tensor<3x3xf32> {
  %shape = arith.constant dense<[3, 3]>  : tensor<2xi32>
  %0 = "stablehlo.rng"(%min, %max, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:  %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:  %[[IDX1_CAST:.+]] = arith.index_cast %[[IDX1]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[IDX0_CAST]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL3:.+]] = arith.addi %[[IDX1_CAST]], %[[VAL2]] : i32
// CHECK-DAG:    %[[VAL4:.+]] = arith.muli %[[VAL3]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL5:.+]] = arith.addi %[[VAL4]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL5_CAST:.+]] = arith.uitofp %[[VAL5]] : i32 to f32
// CHECK-DAG:    %[[VAL6:.+]] = arith.mulf %[[VAL5_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL7:.+]] = arith.addf %[[VAL6]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL7]] : f32
// CHECK-NEXT: -> tensor<3x3xf32>

// -----

// CHECK-LABEL: func @rng_uniform_3d
func.func @rng_uniform_3d(%min: tensor<f32>, %max: tensor<f32>) -> tensor<2x2x2xf32> {
  %shape = arith.constant dense<[2, 2, 2]>  : tensor<3xi32>
  %0 = "stablehlo.rng"(%min, %max, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi32>) -> tensor<2x2x2xf32>
  func.return %0 : tensor<2x2x2xf32>
}
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:  %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:  %[[IDX1_CAST:.+]] = arith.index_cast %[[IDX1]] : index to i32
// CHECK-DAG:  %[[IDX2:.+]] = linalg.index 2 : index
// CHECK-DAG:  %[[IDX2_CAST:.+]] = arith.index_cast %[[IDX2]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[IDX0_CAST]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL3:.+]] = arith.addi %[[IDX1_CAST]], %[[VAL2]] : i32
// CHECK-DAG:    %[[VAL4:.+]] = arith.muli %[[VAL3]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL5:.+]] = arith.addi %[[VAL4]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL6:.+]] = arith.addi %[[IDX2_CAST]], %[[VAL5]] : i32
// CHECK-DAG:    %[[VAL7:.+]] = arith.muli %[[VAL6]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL8:.+]] = arith.addi %[[VAL7]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL8_CAST:.+]] = arith.uitofp %[[VAL8]] : i32 to f32
// CHECK-DAG:    %[[VAL6:.+]] = arith.mulf %[[VAL8_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL7:.+]] = arith.addf %[[VAL6]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL7]] : f32
// CHECK-NEXT: -> tensor<2x2x2xf32>

// -----

// CHECK-LABEL: func @rng_uniform_dynamic_1d
func.func @rng_uniform_dynamic_1d(%min: tensor<f32>, %max: tensor<f32>, %shape: tensor<1xi32>) -> tensor<?xf32> {
  %0 = "stablehlo.rng"(%min, %max, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[EX:.+]] = tensor.extract %{{.*}}[%[[C0]]]
// CHECK-DAG:    %[[IND:.+]] = arith.index_cast %[[EX]] : i32 to index
// CHECK-DAG:    %{{.+}} = tensor.empty(%[[IND]]) : tensor<?xf32>
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:    %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:    %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[IDX0_CAST]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL2_CAST:.+]] = arith.uitofp %[[VAL2]] : i32 to f32
// CHECK-DAG:    %[[VAL4:.+]] = arith.mulf %[[VAL2_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL5:.+]] = arith.addf %[[VAL4]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL5]] : f32
// CHECK-NEXT: -> tensor<?xf32>

// -----

// CHECK-LABEL: func.func @three_fry_i64(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2xi64>
func.func @three_fry_i64(%arg0: tensor<2xi64>) -> (tensor<2xi64>, tensor<8xi64>) {
  %output_state, %output = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>} : (tensor<2xi64>) -> (tensor<2xi64>, tensor<8xi64>)
  return %output_state, %output : tensor<2xi64>, tensor<8xi64>
}
// CHECK-DAG: %[[VAL_1:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[VAL_2:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[VAL_4:.*]] = arith.constant 8 : i32
// CHECK-DAG: %[[VAL_5:.*]] = arith.constant 24 : i32
// CHECK-DAG: %[[VAL_6:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[VAL_7:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[VAL_8:.*]] = arith.constant 29 : i32
// CHECK-DAG: %[[VAL_9:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[VAL_10:.*]] = arith.constant 6 : i32
// CHECK-DAG: %[[VAL_11:.*]] = arith.constant 26 : i32
// CHECK-DAG: %[[VAL_12:.*]] = arith.constant 17 : i32
// CHECK-DAG: %[[VAL_13:.*]] = arith.constant 15 : i32
// CHECK-DAG: %[[VAL_14:.*]] = arith.constant 19 : i32
// CHECK-DAG: %[[VAL_15:.*]] = arith.constant 13 : i32
// CHECK-DAG: %[[VAL_16:.*]] = arith.constant 466688986 : i32
// CHECK-DAG: %[[VAL_17:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[VAL_18:.*]] = arith.constant 32 : i64
// CHECK-DAG: %[[VAL_19:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[VAL_20:.*]] = arith.constant 1 : index

// CHECK-DAG: %[[VAL_21:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_20]]] : tensor<2xi64>
// CHECK-DAG: %[[VAL_22:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_19]]] : tensor<2xi64>
// CHECK-DAG: %[[VAL_23:.*]] = arith.trunci %[[VAL_22]] : i64 to i32
// CHECK-DAG: %[[VAL_24:.*]] = arith.shrui %[[VAL_22]], %[[VAL_18]] : i64
// CHECK-DAG: %[[VAL_25:.*]] = arith.trunci %[[VAL_24]] : i64 to i32
// CHECK-DAG: %[[VAL_26:.*]] = arith.addi %[[VAL_21]], %[[VAL_17]] : i64
// CHECK-DAG: %[[VAL_27:.*]] = tensor.empty() : tensor<8xi64>
// CHECK-DAG: %[[VAL_28:.*]] = arith.xori %[[VAL_23]], %[[VAL_16]] : i32
// CHECK-DAG: %[[VAL_29:.*]] = arith.xori %[[VAL_28]], %[[VAL_25]] : i32

// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: {indexing_maps = [#map], iterator_types = ["parallel"]}
// CHECK-SAME: outs(%[[VAL_27]] : tensor<8xi64>)

// CHECK: ^bb0(%[[VAL_31:.*]]: i64):

// CHECK-DAG:   %[[VAL_32:.*]] = linalg.index 0 : index
// CHECK-DAG:   %[[VAL_33:.*]] = arith.index_cast %[[VAL_32]] : index to i64
// CHECK-DAG:   %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_21]] : i64
// CHECK-DAG:   %[[VAL_35:.*]] = arith.trunci %[[VAL_34]] : i64 to i32
// CHECK-DAG:   %[[VAL_36:.*]] = arith.shrui %[[VAL_34]], %[[VAL_18]] : i64
// CHECK-DAG:   %[[VAL_37:.*]] = arith.trunci %[[VAL_36]] : i64 to i32

// CHECK-DAG:   %[[VAL_38:.*]] = arith.addi %[[VAL_35]], %[[VAL_23]] : i32
// CHECK-DAG:   %[[VAL_39:.*]] = arith.addi %[[VAL_37]], %[[VAL_25]] : i32

// CHECK-DAG:   %[[VAL_40:.*]] = arith.addi %[[VAL_38]], %[[VAL_39]] : i32
// CHECK-DAG:   %[[VAL_41:.*]] = arith.shli %[[VAL_39]], %[[VAL_15]] : i32
// CHECK-DAG:   %[[VAL_42:.*]] = arith.shrui %[[VAL_39]], %[[VAL_14]] : i32
// CHECK-DAG:   %[[VAL_43:.*]] = arith.ori %[[VAL_41]], %[[VAL_42]] : i32
// CHECK-DAG:   %[[VAL_44:.*]] = arith.xori %[[VAL_40]], %[[VAL_43]] : i32

// CHECK-DAG:   %[[VAL_45:.*]] = arith.addi %[[VAL_40]], %[[VAL_44]] : i32
// CHECK-DAG:   %[[VAL_46:.*]] = arith.shli %[[VAL_44]], %[[VAL_13]] : i32
// CHECK-DAG:   %[[VAL_47:.*]] = arith.shrui %[[VAL_44]], %[[VAL_12]] : i32
// CHECK-DAG:   %[[VAL_48:.*]] = arith.ori %[[VAL_46]], %[[VAL_47]] : i32
// CHECK-DAG:   %[[VAL_49:.*]] = arith.xori %[[VAL_45]], %[[VAL_48]] : i32

// CHECK-DAG:   %[[VAL_50:.*]] = arith.addi %[[VAL_45]], %[[VAL_49]] : i32
// CHECK-DAG:   %[[VAL_51:.*]] = arith.shli %[[VAL_49]], %[[VAL_11]] : i32
// CHECK-DAG:   %[[VAL_52:.*]] = arith.shrui %[[VAL_49]], %[[VAL_10]] : i32
// CHECK-DAG:   %[[VAL_53:.*]] = arith.ori %[[VAL_51]], %[[VAL_52]] : i32
// CHECK-DAG:   %[[VAL_54:.*]] = arith.xori %[[VAL_50]], %[[VAL_53]] : i32

// CHECK-DAG:   %[[VAL_55:.*]] = arith.addi %[[VAL_50]], %[[VAL_54]] : i32
// CHECK-DAG:   %[[VAL_56:.*]] = arith.shli %[[VAL_54]], %[[VAL_10]] : i32
// CHECK-DAG:   %[[VAL_57:.*]] = arith.shrui %[[VAL_54]], %[[VAL_11]] : i32
// CHECK-DAG:   %[[VAL_58:.*]] = arith.ori %[[VAL_56]], %[[VAL_57]] : i32
// CHECK-DAG:   %[[VAL_59:.*]] = arith.xori %[[VAL_55]], %[[VAL_58]] : i32

// CHECK-DAG:   %[[VAL_60:.*]] = arith.addi %[[VAL_55]], %[[VAL_25]] : i32
// CHECK-DAG:   %[[VAL_61:.*]] = arith.addi %[[VAL_59]], %[[VAL_29]] : i32
// CHECK-DAG:   %[[VAL_62:.*]] = arith.addi %[[VAL_61]], %[[VAL_9]] : i32
// CHECK-DAG:   %[[VAL_63:.*]] = arith.addi %[[VAL_60]], %[[VAL_62]] : i32
// CHECK-DAG:   %[[VAL_64:.*]] = arith.shli %[[VAL_62]], %[[VAL_12]] : i32
// CHECK-DAG:   %[[VAL_65:.*]] = arith.shrui %[[VAL_62]], %[[VAL_13]] : i32
// CHECK:   %[[VAL_66:.*]] = arith.ori %[[VAL_64]], %[[VAL_65]] : i32

// CHECK:   linalg.yield %[[YIELDED:.*]] : i64

// Set the updated state.
// CHECK: %[[VAL_159:.*]] = tensor.insert %[[VAL_26]] into %[[ARG0]]{{\[}}%[[VAL_20]]] : tensor<2xi64>

// CHECK: return %[[VAL_159]], %[[GENERIC:.*]] : tensor<2xi64>, tensor<8xi64>

// -----

// CHECK-LABEL: func.func @three_fry_i32
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2xi64>
func.func @three_fry_i32(%arg0: tensor<2xi64>) -> (tensor<2xi64>, tensor<8xi32>) {
  %output_state, %output = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>} : (tensor<2xi64>) -> (tensor<2xi64>, tensor<8xi32>)
  return %output_state, %output : tensor<2xi64>, tensor<8xi32>
}
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64

// Check we update state correctly:
// CHECK: %[[STATE:.+]] = tensor.extract %[[ARG0]][%[[C1]]] : tensor<2xi64>
// CHECK: %[[NEWSTATE:.+]] = arith.addi %[[STATE]], %[[C4]] : i64

// CHECK: %[[DEST0:.+]] = tensor.empty() : tensor<4xi32>
// CHECK: %[[DEST1:.+]] = tensor.empty() : tensor<4xi32>
// CHECK: %[[GENERIC:.+]]:2 = linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map]
// CHECK-SAME: iterator_types = ["parallel"]}
// CHECK-SAME: outs(%[[DEST0]], %[[DEST1]] : tensor<4xi32>, tensor<4xi32>)

// CHECK: %expanded = tensor.expand_shape %[[GENERIC]]#0
// CHECK-SAME{literal}: [[0, 1]] : tensor<4xi32> into tensor<4x1xi32>

// CHECK: %expanded_1 = tensor.expand_shape %[[GENERIC]]#1
// CHECK-SAME{literal}: [[0, 1]] : tensor<4xi32> into tensor<4x1xi32>

// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4x2xi32>
// CHECK: %[[CONCAT:.+]] = linalg.generic
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x2xi32>)

// CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[CONCAT]]
// CHECK-SAME{literal}: [[0, 1]] : tensor<4x2xi32> into tensor<8xi32>
// CHECK: %[[INSERTED:.+]] = tensor.insert %[[NEWSTATE]] into %[[ARG0]][%[[C1]]] : tensor<2xi64>

// CHECK: return %[[INSERTED]], %[[COLLAPSE]]


// -----

// CHECK-LABEL: func.func @three_fry_odd_i32
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2xi64>
func.func @three_fry_odd_i32(%arg0: tensor<2xi64>) -> (tensor<2xi64>, tensor<7x11xi32>) {
  %output_state, %output = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>} : (tensor<2xi64>) -> (tensor<2xi64>, tensor<7x11xi32>)
  return %output_state, %output : tensor<2xi64>, tensor<7x11xi32>
}
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C42:.+]] = arith.constant 42 : i64

// Check we update state correctly:
// CHECK: %[[STATE:.+]] = tensor.extract %[[ARG0]][%[[C1]]] : tensor<2xi64>
// CHECK: %[[NEWSTATE:.+]] = arith.addi %[[STATE]], %[[C42]] : i64

// CHECK: %[[DEST0:.+]] = tensor.empty() : tensor<42xi32>
// CHECK: %[[DEST1:.+]] = tensor.empty() : tensor<42xi32>
// CHECK: %[[GENERIC:.+]]:2 = linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map]
// CHECK-SAME: iterator_types = ["parallel"]}
// CHECK-SAME: outs(%[[DEST0]], %[[DEST1]] : tensor<42xi32>, tensor<42xi32>)

// CHECK: %expanded = tensor.expand_shape %[[GENERIC]]#0
// CHECK-SAME{literal}: [[0, 1]] : tensor<4xi32> into tensor<7x6x1xi32>

// CHECK: %expanded_1 = tensor.expand_shape %[[GENERIC]]#1
// CHECK-SAME{literal}: [[0, 1]] : tensor<4xi32> into tensor<7x6x1xi32>

// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<7x6x2xi32>
// CHECK: %[[CONCAT:.+]] = linalg.generic
// CHECK-SAME: outs(%[[EMPTY]] : tensor<7x6x2xi32>)

// CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %10
// CHECK-SAME{literal}: [[0], [1, 2]] : tensor<7x6x2xi32> into tensor<7x12xi32>

// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[COLLAPSE]][0, 0] [7, 11] [1, 1]
// CHECK: %[[INSERTED:.+]] = tensor.insert %[[NEWSTATE]] into %[[ARG0]][%[[C1]]] : tensor<2xi64>
// CHECK: return %[[INSERTED]], %[[SLICE]] : tensor<2xi64>, tensor<7x11xi32>

// -----

// CHECK-LABEL: func.func @three_fry_i16
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2xi64>
func.func @three_fry_i16(%arg0: tensor<2xi64>) -> (tensor<2xi64>, tensor<8xi16>) {
  %output_state, %output = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>} : (tensor<2xi64>) -> (tensor<2xi64>, tensor<8xi16>)
  return %output_state, %output : tensor<2xi64>, tensor<8xi16>
}
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64

// Check we update state correctly:
// CHECK: %[[STATE:.+]] = tensor.extract %[[ARG0]][%[[C1]]] : tensor<2xi64>
// CHECK: %[[NEWSTATE:.+]] = arith.addi %[[STATE]], %[[C4]] : i64

// CHECK: %[[DEST0:.+]] = tensor.empty() : tensor<4xi16>
// CHECK: %[[DEST1:.+]] = tensor.empty() : tensor<4xi16>
// CHECK: %[[GENERIC:.+]]:2 = linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map]
// CHECK-SAME: iterator_types = ["parallel"]}
// CHECK-SAME: outs(%[[DEST0]], %[[DEST1]] : tensor<4xi16>, tensor<4xi16>)

// CHECK: %expanded = tensor.expand_shape %[[GENERIC]]#0
// CHECK-SAME{literal}: [[0, 1]] : tensor<4xi16> into tensor<4x1xi16>

// CHECK: %expanded_1 = tensor.expand_shape %[[GENERIC]]#1
// CHECK-SAME{literal}: [[0, 1]] : tensor<4xi16> into tensor<4x1xi16>

// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[CONCAT:.+]] = linalg.generic
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x2xi16>)

// CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[CONCAT]]
// CHECK-SAME{literal}: [[0, 1]] : tensor<4x2xi16> into tensor<8xi16>
// CHECK: %[[INSERTED:.+]] = tensor.insert %[[NEWSTATE]] into %[[ARG0]][%[[C1]]] : tensor<2xi64>

// CHECK: return %[[INSERTED]], %[[COLLAPSE]] : tensor<2xi64>, tensor<8xi16>
