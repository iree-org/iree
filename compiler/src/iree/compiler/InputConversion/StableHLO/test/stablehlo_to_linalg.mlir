// RUN: iree-opt %s --iree-stablehlo-to-linalg --split-input-file \
// RUN:   --canonicalize | FileCheck %s

// CHECK-LABEL: func @bitcast_convert
func.func @bitcast_convert(%input: tensor<2x2xi32>) -> tensor<2x2xf32> {
  %result = "stablehlo.bitcast_convert"(%input) : (tensor<2x2xi32>) -> tensor<2x2xf32>
  func.return %result : tensor<2x2xf32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.bitcast %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.bitcast

// -----

// CHECK-LABEL: func @bitcast_convert_dynamic
func.func @bitcast_convert_dynamic(%input: tensor<?x?xi32>) -> tensor<?x?xf32> {
  %result = "stablehlo.bitcast_convert"(%input) : (tensor<?x?xi32>) -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.bitcast %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.bitcast

// -----

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @bitcast_convert_expand
func.func @bitcast_convert_expand(%input: tensor<6xi32>) -> tensor<6x4xi8> {
  %result = "stablehlo.bitcast_convert"(%input) : (tensor<6xi32>) -> tensor<6x4xi8>
  func.return %result : tensor<6x4xi8>
}

// CHECK: %[[C8:.*]] = arith.constant 8 : i32
// CHECK: tensor.empty() : tensor<6x4xi8>
// CHECK: %[[RESULT:.*]] = linalg.generic {
// CHECK:    indexing_maps = [#[[MAP0]], #[[MAP1]]],
// CHECK:    iterator_types = ["parallel", "parallel"]}
// CHECK:    ^bb0(%[[IN:.*]]: i32, %[[OUT:.*]]: i8):
// CHECK:      %[[IOTA:.*]] = linalg.index 1 : index
// CHECK:      %[[IOTA_CASTED:.*]] = arith.index_cast %[[IOTA]] : index to i32
// CHECK:      %[[AMT:.*]] = arith.muli %[[IOTA_CASTED]], %[[C8]] : i32
// CHECK:      %[[SHIFT:.*]] = arith.shrui %[[IN]], %[[AMT]] : i32
// CHECK:      %[[TRUNC:.*]] = arith.trunci %[[SHIFT]] : i32 to i8
// CHECK:      linalg.yield %[[TRUNC]] : i8
// CHECK:    } -> tensor<6x4xi8>
// CHECK:    return %[[RESULT]] : tensor<6x4xi8>

// -----

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: func @bitcast_convert_contract
func.func @bitcast_convert_contract(%input: tensor<7x4xi8>) -> tensor<7xi32> {
  %result = "stablehlo.bitcast_convert"(%input) : (tensor<7x4xi8>) -> tensor<7xi32>
  func.return %result : tensor<7xi32>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<7xi32>
// CHECK: linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<7xi32>) -> tensor<7xi32>
// CHECK: %[[RESULT:.*]] = linalg.generic {
// CHECK:    indexing_maps = [#[[MAP0]], #[[MAP1]]],
// CHECK:    iterator_types = ["parallel", "reduction"]}
// CHECK:    ^bb0(%[[IN:.*]]: i8, %[[OUT:.*]]: i32):
// CHECK:      %[[IOTA:.*]] = linalg.index 1 : index
// CHECK:      %[[IOTA_CASTED:.*]] = arith.index_cast %[[IOTA]] : index to i32
// CHECK:      %[[AMT:.*]] = arith.muli %[[IOTA_CASTED]], %[[C8]] : i3
// CHECK:      %[[EXT:.*]] = arith.extui %[[IN]] : i8 to i32
// CHECK:      %[[SHIFT:.*]] = arith.shli %[[EXT]], %[[AMT]] : i32
// CHECK:      %[[OR:.*]] = arith.ori %[[SHIFT]], %[[OUT]] : i32
// CHECK:      linalg.yield %[[OR]] : i32
// CHECK: } -> tensor<7xi32>
// CHECK: return %[[RESULT]] : tensor<7xi32>

// -----

// CHECK-LABEL:   func @concatenate(
// CHECK-SAME:   %[[VAL_0:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_1:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_2:[a-zA-Z0-9_]*]]
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?x?xi32>
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_1]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_2]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : index
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_15]] : index
// CHECK:           %[[VAL_23:.*]] = tensor.empty(%[[VAL_5]], %[[VAL_17]]) : tensor<?x?xi32>
// CHECK:           %[[VAL_24:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%[[VAL_23]] : tensor<?x?xi32>) {
// CHECK:           ^bb0(%[[VAL_25:.*]]: i32):
// CHECK:             %[[VAL_26:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_28:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_30:.*]] = tensor.dim %[[VAL_0]], %[[C1]] : tensor<?x?xi32>
// CHECK:             %[[VAL_32:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_30]] : index
// CHECK:             %[[VAL_33:.*]] = scf.if %[[VAL_32]] -> (i32) {
// CHECK:               %[[VAL_35:.*]] = tensor.extract %[[VAL_0]][%[[VAL_26]], %[[VAL_28]]] : tensor<?x?xi32>
// CHECK:               scf.yield %[[VAL_35]] : i32
// CHECK:             } else {
// CHECK:               %[[VAL_37:.*]] = tensor.dim %[[VAL_1]], %[[C1]] : tensor<?x?xi32>
// CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_30]], %[[VAL_37]] : index
// CHECK:               %[[VAL_39:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_38]] : index
// CHECK:               %[[VAL_40:.*]] = scf.if %[[VAL_39]] -> (i32) {
// CHECK:                 %[[VAL_41:.*]] = arith.subi %[[VAL_28]], %[[VAL_30]] : index
// CHECK:                 %[[VAL_42:.*]] = tensor.extract %[[VAL_1]][%[[VAL_26]], %[[VAL_41]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_42]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_43:.*]] = arith.subi %[[VAL_28]], %[[VAL_38]] : index
// CHECK:                 %[[VAL_44:.*]] = tensor.extract %[[VAL_2]][%[[VAL_26]], %[[VAL_43]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_44]] : i32
// CHECK:               }
// CHECK:               scf.yield %[[VAL_45:.*]] : i32
// CHECK:             }
// CHECK:             linalg.yield %[[VAL_46:.*]] : i32
// CHECK:           } -> tensor<?x?xi32>
// CHECK:           return %[[VAL_47:.*]] : tensor<?x?xi32>
// CHECK:         }
func.func @concatenate(%a: tensor<?x?xi32>, %b: tensor<?x?xi32>, %c: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %concat = "stablehlo.concatenate"(%a, %b, %c) {
      dimension = 1
    } : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    func.return %concat : tensor<?x?xi32>
}

// -----

// CHECK-LABEL:   func @concatenate_unsigned(
// CHECK-SAME:   %[[VAL_0:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_1:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_2:[a-zA-Z0-9_]*]]
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK-DAG:       %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK-DAG:       %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_3]], %[[C0]] : tensor<?x?xi32>
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_3]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_4]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_18:.*]] = tensor.dim %[[VAL_5]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_10]], %[[VAL_14]] : index
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_18]] : index
// CHECK:           %[[VAL_26:.*]] = tensor.empty(%[[VAL_8]], %[[VAL_20]]) : tensor<?x?xi32>
// CHECK:           %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%[[VAL_26]] : tensor<?x?xi32>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: i32):
// CHECK:             %[[VAL_29:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_30:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_33:.*]] = tensor.dim %[[VAL_3]], %[[C1]] : tensor<?x?xi32>
// CHECK:             %[[VAL_35:.*]] = arith.cmpi ult, %[[VAL_30]], %[[VAL_33]] : index
// CHECK:             %[[VAL_36:.*]] = scf.if %[[VAL_35]] -> (i32) {
// CHECK:               %[[VAL_38:.*]] = tensor.extract %[[VAL_3]][%[[VAL_29]], %[[VAL_30]]] : tensor<?x?xi32>
// CHECK:               scf.yield %[[VAL_38]] : i32
// CHECK:             } else {
// CHECK:               %[[VAL_40:.*]] = tensor.dim %[[VAL_4]], %[[C1]] : tensor<?x?xi32>
// CHECK:               %[[VAL_41:.*]] = arith.addi %[[VAL_33]], %[[VAL_40]] : index
// CHECK:               %[[VAL_42:.*]] = arith.cmpi ult, %[[VAL_30]], %[[VAL_41]] : index
// CHECK:               %[[VAL_43:.*]] = scf.if %[[VAL_42]] -> (i32) {
// CHECK:                 %[[VAL_44:.*]] = arith.subi %[[VAL_30]], %[[VAL_33]] : index
// CHECK:                 %[[VAL_45:.*]] = tensor.extract %[[VAL_4]][%[[VAL_29]], %[[VAL_44]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_45]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_46:.*]] = arith.subi %[[VAL_30]], %[[VAL_41]] : index
// CHECK:                 %[[VAL_47:.*]] = tensor.extract %[[VAL_5]][%[[VAL_29]], %[[VAL_46]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_47]] : i32
// CHECK:               }
// CHECK:               scf.yield %[[VAL_48:.*]] : i32
// CHECK:             }
// CHECK:             linalg.yield %[[VAL_49:.*]] : i32
// CHECK:           } -> tensor<?x?xi32>
// CHECK:           %[[VAL_50:.*]] = builtin.unrealized_conversion_cast %[[VAL_51:.*]] : tensor<?x?xi32> to tensor<?x?xui32>
// CHECK:           return %[[VAL_50]] : tensor<?x?xui32>
// CHECK:         }
func.func @concatenate_unsigned(%a: tensor<?x?xui32>, %b: tensor<?x?xui32>, %c: tensor<?x?xui32>) -> tensor<?x?xui32> {
    %concat = "stablehlo.concatenate"(%a, %b, %c) {
      dimension = 1
    } : (tensor<?x?xui32>, tensor<?x?xui32>, tensor<?x?xui32>) -> tensor<?x?xui32>
    func.return %concat : tensor<?x?xui32>
}

// -----

// CHECK-LABEL: func @constant
// CHECK: %[[CONSTANT:.*]] = arith.constant dense<10> : tensor<i32>
func.func @constant() -> tensor<i32> {
  %result = "stablehlo.constant"() {
    value = dense<10> : tensor<i32>
  } : () -> (tensor<i32>)
  func.return %result : tensor<i32>
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: func @einsum_basic
// CHECK-SAME:  (%[[LHS:.*]]: tensor<3x4x5xf32>, %[[RHS:.*]]: tensor<3x5x6xf32>)
func.func @einsum_basic(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ijk,ikm->ijm", someattr}: (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>
}
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<3x4x6xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<3x4x5xf32>, tensor<3x5x6xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<3x4x6xf32>)
// CHECK-SAME: {someattr}
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @einsum_pointwisemul
func.func @einsum_pointwisemul(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "abc,abc->abc"} : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  func.return %0 : tensor<3x4x5xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<3x4x5xf32>, %[[RHS:.*]]: tensor<3x4x5xf32>)
// CHECK: tensor.empty() : tensor<3x4x5xf32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP0]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<3x4x5xf32>, tensor<3x4x5xf32>)
// CHECK-SAME: outs(%[[DST:.+]] : tensor<3x4x5xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[RES:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: func @einsum_matmul
func.func @einsum_matmul(%arg0: tensor<7x9xf32>, %arg1: tensor<9x5xf32>) -> tensor<7x5xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ae,ed->ad"}: (tensor<7x9xf32>, tensor<9x5xf32>) -> tensor<7x5xf32>
  func.return %0 : tensor<7x5xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<7x9xf32>, %[[RHS:.*]]: tensor<9x5xf32>)
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<7x5xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<7x9xf32>, tensor<9x5xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<7x5xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>
// CHECK: func @einsum_broadcast4
func.func @einsum_broadcast4(%arg0: tensor<3x4x5x6x7xf32>, %arg1: tensor<7x8xf32>) -> tensor<3x4x5x6x8xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "abcdh,hg->abcdg"}: (tensor<3x4x5x6x7xf32>, tensor<7x8xf32>) -> tensor<3x4x5x6x8xf32>
  func.return %0 : tensor<3x4x5x6x8xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<3x4x5x6x7xf32>, %[[RHS:.*]]: tensor<7x8xf32>)
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<3x4x5x6x8xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<3x4x5x6x7xf32>, tensor<7x8xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<3x4x5x6x8xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: func @einsum_ellipsis
func.func @einsum_ellipsis(%arg0: tensor<1x512x128xf32>, %arg1: tensor<128x256xf32>) -> tensor<1x512x256xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "...x,xy->...y"} : (tensor<1x512x128xf32>, tensor<128x256xf32>) -> tensor<1x512x256xf32>
  func.return %0 : tensor<1x512x256xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<1x512x128xf32>, %[[RHS:.*]]: tensor<128x256xf32>)
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<1x512x256xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<1x512x128xf32>, tensor<128x256xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<1x512x256xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: func @einsum_dynamic_size_broadcast_dot
func.func @einsum_dynamic_size_broadcast_dot(%arg0: tensor<?x?x4xf32>, %arg1: tensor<4x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "abc,cd->abd"} : (tensor<?x?x4xf32>, tensor<4x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<?x?x4xf32>, %[[RHS:.*]]: tensor<4x?xf32>)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[DIM0:.+]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?x4xf32>
// CHECK: %[[DIM1:.+]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?x4xf32>
// CHECK: %[[DIM2:.+]] = tensor.dim %[[RHS]], %[[C1:.+]] : tensor<4x?xf32>
// CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]]) : tensor<?x?x?xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<?x?x4xf32>, tensor<4x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, 0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func @broadcast_in_dim
func.func @broadcast_in_dim(%operand: tensor<5x7x1xf32>) -> tensor<7x10x6x4x5xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[4,0,2]> : tensor<3xi64>}
         : (tensor<5x7x1xf32>) -> tensor<7x10x6x4x5xf32>
  func.return %0 : tensor<7x10x6x4x5xf32>
}
// CHECK: tensor.empty() : tensor<7x10x6x4x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim
// CHECK-PRIMITIVE: tensor.collapse_shape
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE:   permutation = [1, 0]
// CHECK-PRIMITIVE: tensor.empty() : tensor<7x10x6x4x5xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [1, 2, 3]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, 0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func @broadcast_in_dim_ui32
func.func @broadcast_in_dim_ui32(%operand: tensor<5x7x1xui32>) -> tensor<7x10x6x4x5xui32> {
  %0 = "stablehlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[4,0,2]> : tensor<3xi64>}
         : (tensor<5x7x1xui32>) -> tensor<7x10x6x4x5xui32>
  func.return %0 : tensor<7x10x6x4x5xui32>
}
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<5x7x1xui32> to tensor<5x7x1xi32>
// CHECK: tensor.empty() : tensor<7x10x6x4x5xi32>
// CHECK: %[[RES:.*]] = linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : i32
// CHECK: builtin.unrealized_conversion_cast %[[RES]] : tensor<7x10x6x4x5xi32> to tensor<7x10x6x4x5xui32>

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_ui32
// CHECK-PRIMITIVE: tensor.collapse_shape
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE:   permutation = [1, 0]
// CHECK-PRIMITIVE: tensor.empty() : tensor<7x10x6x4x5xi32>
// CHECK-PRIMITIVE: %[[RES:.*]] = linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [1, 2, 3]
// CHECK-PRIMITIVE: builtin.unrealized_conversion_cast %[[RES]] : tensor<7x10x6x4x5xi32> to tensor<7x10x6x4x5xui32>

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @broadcast_in_dim_with_one_to_one
func.func @broadcast_in_dim_with_one_to_one(
         %operand: tensor<1xf32>) -> tensor<1x5xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
         : (tensor<1xf32>) -> tensor<1x5xf32>
  func.return %0 : tensor<1x5xf32>
}
// CHECK: tensor.empty() : tensor<1x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_with_one_to_one
// CHECK-PRIMITIVE-NOT: tensor.collapse_shape
// CHECK-PRIMITIVE-NOT: linalg.transpose
// CHECK-PRIMITIVE:     linalg.broadcast
// CHECK-PRIMITIVE:       dimensions = [1]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func @broadcast_in_dim_with_transpose
func.func @broadcast_in_dim_with_transpose(
         %operand: tensor<2x3x4xf32>) -> tensor<3x4x2x5xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[2, 0, 1]> : tensor<3xi64>}
         : (tensor<2x3x4xf32>) -> tensor<3x4x2x5xf32>
  func.return %0 : tensor<3x4x2x5xf32>
}
// CHECK: tensor.empty() : tensor<3x4x2x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_with_transpose
// CHECK-PRIMITIVE: tensor.empty() : tensor<3x4x2xf32>
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE:   permutation = [1, 2, 0]
// CHECK-PRIMITIVE: tensor.empty() : tensor<3x4x2x5xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [3]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @broadcast_in_dim_scalar
func.func @broadcast_in_dim_scalar(%operand: tensor<f32>) -> tensor<7x10x6xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%operand)
        {broadcast_dimensions = dense<[]> : tensor<0xi64>}
        : (tensor<f32>) -> tensor<7x10x6xf32>
  func.return %0 : tensor<7x10x6xf32>
}
// CHECK: tensor.empty() : tensor<7x10x6xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_scalar
// CHECK-PRIMITIVE: tensor.empty() : tensor<7x10x6xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [0, 1, 2]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @broadcast_scalar
func.func @broadcast_scalar(%arg: tensor<f32>) -> tensor<4x2x1xf32> {
  %0 = "stablehlo.broadcast"(%arg) {broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>} : (tensor<f32>) -> tensor<4x2x1xf32>
  func.return %0: tensor<4x2x1xf32>
}
// CHECK: tensor.empty() : tensor<4x2x1xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_scalar
// CHECK-PRIMITIVE: tensor.empty() : tensor<4x2x1xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE-SAME: ins(
// CHECK-PRIMITIVE-SAME: outs(
// CHECK-PRIMITIVE-SAME: dimensions = [0, 1, 2]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK: func @broadcast
func.func @broadcast(%arg: tensor<4x?x16xf32>) -> tensor<4x2x1x4x?x16xf32> {
  %0 = "stablehlo.broadcast"(%arg) {broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>} : (tensor<4x?x16xf32>) -> tensor<4x2x1x4x?x16xf32>
  func.return %0: tensor<4x2x1x4x?x16xf32>
}
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[C1]] : tensor<4x?x16xf32>
// CHECK: %{{.*}} = tensor.empty(%[[DIM]]) : tensor<4x2x1x4x?x16xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast
// CHECK-PRIMITIVE-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-PRIMITIVE: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[C1]] : tensor<4x?x16xf32>
// CHECK-PRIMITIVE: %{{.*}} = tensor.empty(%[[DIM]]) : tensor<4x2x1x4x?x16xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [0, 1, 2]

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @iota_f32
func.func @iota_f32() -> tensor<7x10xf32> {
  %result = "stablehlo.iota"() {iota_dimension = 1 : i64, someattr} : () -> (tensor<7x10xf32>)
  func.return %result : tensor<7x10xf32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%{{.*}}: f32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : f32

// CHECK-PRIMITIVE-LABEL: func @iota_f32
// CHECK-PRIMITIVE: %[[EMPTY:.*]] = tensor.empty()
// CHECK-PRIMITIVE: linalg.map outs(%[[EMPTY]] : tensor<7x10xf32>
// CHECK-PRIMITIVE-SAME: {someattr}
// CHECK-PRIMITIVE:        %[[INDEX:.*]] = linalg.index 1
// CHECK-PRIMITIVE-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i64
// CHECK-PRIMITIVE-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i64 to f32
// CHECK-PRIMITIVE-NEXT:   linalg.yield %[[FLOAT_CAST]]

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @iota_i32
func.func @iota_i32() -> tensor<7x10xi32> {
  %result = "stablehlo.iota"() {iota_dimension = 1 : i64} : () -> (tensor<7x10xi32>)
  func.return %result : tensor<7x10xi32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   linalg.yield %[[INT_CAST]] : i32

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @iota_ui32
func.func @iota_ui32() -> tensor<7x10xui32> {
  %result = "stablehlo.iota"() {iota_dimension = 1 : i64} : () -> (tensor<7x10xui32>)
  func.return %result : tensor<7x10xui32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   linalg.yield %[[INT_CAST]] : i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<7x10xi32> to tensor<7x10xui32>

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @iota_complexf32
func.func @iota_complexf32() -> tensor<7x10xcomplex<f32>> {
  %result = "stablehlo.iota"() {iota_dimension = 1 : i64} : () -> (tensor<7x10xcomplex<f32>>)
  func.return %result : tensor<7x10xcomplex<f32>>
}
// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: complex<f32>):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   %[[COMPLEX_CAST:.*]] = complex.create %[[FLOAT_CAST]], %[[ZERO]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[COMPLEX_CAST]] : complex<f32>

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @dynamic_iota_f32
// CHECK-SAME: %[[SHAPE:.*]]: tensor<?xi32>
func.func @dynamic_iota_f32(%shape: tensor<?xi32>) -> tensor<?x?x8xf32> {
  %result = "stablehlo.dynamic_iota"(%shape) {iota_dimension = 1 : i64} : (tensor<?xi32>) -> (tensor<?x?x8xf32>)
  func.return %result : tensor<?x?x8xf32>
}
// CHECK: %[[V1:.*]] = tensor.extract %[[SHAPE]][%c0]
// CHECK: %[[I1:.*]] = arith.index_cast %[[V1]] : i32 to index
// CHECK: %[[V2:.*]] = tensor.extract %[[SHAPE]][%c1]
// CHECK: %[[I2:.*]] = arith.index_cast %[[V2]] : i32 to index
// CHECK: tensor.empty(%[[I1]], %[[I2]]) : tensor<?x?x8xf32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: f32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : f32

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @dyanmic_iota_ui32
// CHECK-SAME: %[[SHAPE:.*]]: tensor<?xi32>
func.func @dyanmic_iota_ui32(%shape: tensor<?xi32>) -> tensor<?x?x8xui32> {
  %result = "stablehlo.dynamic_iota"(%shape) {iota_dimension = 1 : i64} : (tensor<?xi32>) -> (tensor<?x?x8xui32>)
  func.return %result : tensor<?x?x8xui32>
}
// CHECK: %[[V1:.*]] = tensor.extract %[[SHAPE]][%c0]
// CHECK: %[[I1:.*]] = arith.index_cast %[[V1]] : i32 to index
// CHECK: %[[V2:.*]] = tensor.extract %[[SHAPE]][%c1]
// CHECK: %[[I2:.*]] = arith.index_cast %[[V2]] : i32 to index
// CHECK: tensor.empty(%[[I1]], %[[I2]]) : tensor<?x?x8xi32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<?x?x8xi32> to tensor<?x?x8xui32>

// -----

func.func @map_mixed(%arg0: tensor<?xf32>,
                     %arg1: tensor<4xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>}
  : (tensor<?xf32>, tensor<4xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @map_mixed
// CHECK: linalg.generic

// CHECK-PRIMITIVE-LABEL: @map_mixed
// CHECK-PRIMITIVE: linalg.map

// -----

func.func @map_one_arg(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.map"(%arg0) ({
  ^bb0(%arg2: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg2 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>}
  : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @map_one_arg
// CHECK: linalg.generic

// CHECK-PRIMITIVE-LABEL: @map_one_arg
// CHECK-PRIMITIVE: linalg.map

// -----

// CHECK-LABEL: @reduce_add_unranked
// CHECK-PRIMITIVE-LABEL: @reduce_add_unranked
func.func @reduce_add_unranked(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, someattr} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}
// CHECK: stablehlo.reduce
// CHECK-PRIMITIVE: stablehlo.reduce

// -----

func.func @reduce_dynamic_output(%arg0: tensor<5x4xi32>, %arg1: tensor<i32>) -> tensor<?xi32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = stablehlo.maximum %arg3, %arg4 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5x4xi32>, tensor<i32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// Regression test: just check that this lowers successfully.
// CHECK-LABEL: @reduce_dynamic_output
// CHECK: linalg.generic

// CHECK-PRIMITIVE-LABEL: @reduce_dynamic_output
// CHECK-PRIMITIVE: linalg.reduce

// -----

func.func @pad_cst(%arg0: tensor<12x4xf32>) -> tensor<18x12xf32> {
  %0 = arith.constant dense<0.0> : tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) {
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
  func.return %1 : tensor<18x12xf32>
}
// CHECK-LABEL: func @pad_cst
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
//   CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK: tensor.pad %[[ARG0]] low[4, 5] high[2, 3]
//       CHECK:  tensor.yield %[[CST]] : f32
//       CHECK: } : tensor<12x4xf32> to tensor<18x12xf32>

// -----

func.func @pad_tensor(%arg0: tensor<12x4xf32>, %arg1: tensor<f32>) -> tensor<18x12xf32> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
  func.return %0 : tensor<18x12xf32>
}
// CHECK-LABEL: func @pad_tensor
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
//   CHECK-DAG:   %[[PAD:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
//       CHECK:   tensor.pad %[[ARG0]] low[4, 5] high[2, 3]
//       CHECK:     tensor.yield %[[PAD]] : f32
//       CHECK:   } : tensor<12x4xf32> to tensor<18x12xf32>

// -----

func.func @pad_interior(%arg0: tensor<12x4xui32>, %arg1: tensor<ui32>) -> tensor<29x15xui32> {
  %0 = arith.constant dense<0> : tensor<ui32>
  %1 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
    interior_padding = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<12x4xui32>, tensor<ui32>) -> tensor<29x15xui32>
  func.return %1 : tensor<29x15xui32>
}
// CHECK-LABEL: func @pad_interior
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
//   CHECK-DAG: %[[CAST0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<12x4xui32> to tensor<12x4xi32>
//   CHECK-DAG: %[[CAST1:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : tensor<ui32> to tensor<i32>
//   CHECK-DAG: %[[PAD:.+]] = tensor.extract %[[CAST1]][] : tensor<i32>
//       CHECK: %[[INIT:.+]] = tensor.empty() : tensor<29x15xi32>
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[PAD]] : i32) outs(%[[INIT]] : tensor<29x15xi32>) -> tensor<29x15xi32>
//       CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[CAST0]] into %[[FILL]][4, 5] [12, 4] [2, 2] : tensor<12x4xi32> into tensor<29x15xi32>

// -----

func.func @pad_interior_negative(%arg0: tensor<12x4xui32>, %arg1: tensor<ui32>) -> tensor<25x9xui32> {
  %0 = arith.constant dense<0> : tensor<ui32>
  %1 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<[-2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, -1]> : tensor<2xi64>,
    interior_padding = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<12x4xui32>, tensor<ui32>) -> tensor<25x9xui32>
  func.return %1 : tensor<25x9xui32>
}
// CHECK-LABEL: func @pad_interior_negative
//       CHECK: %[[PAD:.*]] = tensor.insert_slice %{{.+}} into %{{.+}}[4, 0] [12, 4] [2, 2] : tensor<12x4xi32> into tensor<29x10xi32>
//       CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[PAD]][0, 1] [25, 9] [1, 1] : tensor<29x10xi32> to tensor<25x9xi32>

// -----

// CHECK-LABEL: func @real_dynamic_slice
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<256x?xf32>, %[[START_INDICES:.*]]: tensor<2xindex>, %[[LIMIT_INDICES:.*]]: tensor<2xindex>, %[[STRIDES:.*]]: tensor<2xindex>)
func.func @real_dynamic_slice(%input: tensor<256x?xf32>, %start_indices: tensor<2xindex>, %limit_indices: tensor<2xindex>, %strides: tensor<2xindex>) -> tensor<256x?xf32> {
  %0 = "stablehlo.real_dynamic_slice"(%input, %start_indices, %limit_indices, %strides) : (tensor<256x?xf32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<256x?xf32>
  func.return %0 : tensor<256x?xf32>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1

// Fetch start index, limit index and stride.
// CHECK-DAG: %[[START0:.*]] = tensor.extract %[[START_INDICES]][%[[C0]]]
// CHECK-DAG: %[[STRIDE0:.*]] = tensor.extract %[[STRIDES]][%[[C0]]]

// Clamp starting index : 0 <= start <= ub
// CHECK-DAG: %[[MAX0:.*]] = arith.maxsi %[[START0]], %[[C0]] : index
// CHECK-DAG: %[[MIN0:.*]] = arith.minsi %[[MAX0]], %[[C0]] : index

// CHECK-DAG: %[[START1:.*]] = tensor.extract %[[START_INDICES]][%[[C1]]]
// CHECK-DAG: %[[LIMIT1:.*]] = tensor.extract %[[LIMIT_INDICES]][%[[C1]]]
// CHECK-DAG: %[[STRIDE1:.*]] = tensor.extract %[[STRIDES]][%[[C1]]]

// 2.2. Since 1-th dimension of result is unknown we compute result size at 1-th
//      dimension as size[1] = (limit - start)/stride
// CHECK-DAG: %[[DELTA1:.*]] = arith.subi %[[LIMIT1]], %[[START1]] : index
// CHECK-DAG: %[[SIZE1:.*]] = arith.ceildivui %[[DELTA1]], %[[STRIDE1]] : index

// 2.3. Compute upper bound for starting index = operand_dim[1] - size[1].
//      where, size[1] is computed at step 2.2
// CHECK-DAG: %[[OPERAND_DIM1:.*]] = tensor.dim %[[OPERAND]], %[[C1]] : tensor<256x?xf32>
// CHECK-DAG: %[[UB:.*]] = arith.subi %[[OPERAND_DIM1]], %[[SIZE1]] : index

// 2.4. Clamp starting index : 0 <= start <= ub
//      where upper bound (ub) is computed at step 2.3
// CHECK-DAG: %[[MAX1:.*]] = arith.maxsi %[[START1]], %[[C0]] : index
// CHECK-DAG: %[[MIN1:.*]] = arith.minsi %[[MAX1]], %[[UB]] : index

// CHECK-DAG: %[[SLICE:.*]] = tensor.extract_slice %[[OPERAND]][%[[MIN0]], %[[MIN1]]] [256, %[[SIZE1]]] [%[[STRIDE0]], %[[STRIDE1]]] : tensor<256x?xf32> to tensor<256x?xf32>
// CHECK: return %[[SLICE]] : tensor<256x?xf32>

// -----

// Verify that legalization of real_dynamic_slice legalization with integer
// dims work & passes verification.
// CHECK-LABEL: real_dynamic_slice_with_int
func.func @real_dynamic_slice_with_int(%arg0: tensor<10xi32> , %arg1: tensor<1xi32> ) -> tensor<?xi32> {
  %0 = stablehlo.constant dense<0> : tensor<1xi32>
  %1 = stablehlo.constant dense<1> : tensor<1xi32>
  %2 = stablehlo.constant dense<0> : tensor<i32>
  %4 = "stablehlo.real_dynamic_slice"(%arg0, %0, %arg1, %1) : (tensor<10xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  func.return %4 : tensor<?xi32>
}

// -----

// CHECK-LABEL: func @reshape_0D_1D
func.func @reshape_0D_1D(%arg0: tensor<i32>) -> tensor<1xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<i32>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}
// CHECK: tensor.expand_shape %{{.*}} [] : tensor<i32> into tensor<1xi32>

// -----

// CHECK-LABEL: func @reshape_0D_1D_unsigned
// CHECK-SAME:    %[[ARG_UNSIGNED:[a-zA-Z0-9_]*]]
func.func @reshape_0D_1D_unsigned(%arg0: tensor<ui32>) -> tensor<1xui32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<ui32>) -> tensor<1xui32>
  func.return %0 : tensor<1xui32>
}
// CHECK:         %[[ARG_SIGNLESS:.*]] = builtin.unrealized_conversion_cast %[[ARG_UNSIGNED]] : tensor<ui32> to tensor<i32>
// CHECK:         %[[RET_SIGNLESS:.*]] = tensor.expand_shape %[[ARG_SIGNLESS]] [] : tensor<i32> into tensor<1xi32>
// CHECK:         %[[RET_UNSIGNED:.*]] = builtin.unrealized_conversion_cast %[[RET_SIGNLESS]] : tensor<1xi32> to tensor<1xui32>
// CHECK:         return %[[RET_UNSIGNED]] : tensor<1xui32>

// -----

// CHECK-LABEL: func @reshape_1D_0D
func.func @reshape_1D_0D(%arg0: tensor<1xi32>) -> tensor<i32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<1xi32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK: tensor.collapse_shape %{{.*}} [] : tensor<1xi32> into tensor<i32>

// -----

// CHECK-LABEL: func @reshape_1D_0D_unsigned
// CHECK-SAME:    %[[ARG_UNSIGNED:[a-zA-Z0-9_]*]]
func.func @reshape_1D_0D_unsigned(%arg0: tensor<1xui32>) -> tensor<ui32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<1xui32>) -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK:         %[[ARG_SIGNLESS:.*]] = builtin.unrealized_conversion_cast %[[ARG_UNSIGNED]] : tensor<1xui32> to tensor<1xi32>
// CHECK:         %[[RET_SIGNLESS:.*]] = tensor.collapse_shape %[[ARG_SIGNLESS]] [] : tensor<1xi32> into tensor<i32>
// CHECK:         %[[RET_UNSIGNED:.*]] = builtin.unrealized_conversion_cast %[[RET_SIGNLESS]] : tensor<i32> to tensor<ui32>
// CHECK:         return %[[RET_UNSIGNED]] : tensor<ui32>

// -----

// CHECK-LABEL: func @reshape_3D_2D
func.func @reshape_3D_2D(%arg0: tensor<12x1x42xi32>) -> tensor<12x42xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<12x1x42xi32>) -> tensor<12x42xi32>
  func.return %0 : tensor<12x42xi32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2]]

// -----

// CHECK-LABEL: func @reshape_4D_2D
func.func @reshape_4D_2D(%arg0: tensor<12x42x1x1xi32>) -> tensor<12x42xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<12x42x1x1xi32>) -> tensor<12x42xi32>
  func.return %0 : tensor<12x42xi32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2, 3]]

// -----

// CHECK-LABEL: func @reshape_2D_4D
func.func @reshape_2D_4D(%arg0: tensor<12x42xi32>) -> tensor<12x1x42x1xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<12x42xi32>) -> tensor<12x1x42x1xi32>
  func.return %0 : tensor<12x1x42x1xi32>
}
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2, 3]]

// -----

// CHECK-LABEL: func @reshape_3D_4D
func.func @reshape_3D_4D(%arg0: tensor<1x49x16xf32>) -> tensor<1x784x1x1xf32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<1x49x16xf32>) -> tensor<1x784x1x1xf32>
  func.return %0 : tensor<1x784x1x1xf32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2]]
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2, 3]]

// -----

// CHECK-LABEL: func @reshape_4D_3D
func.func @reshape_4D_3D(%arg0: tensor<1x8x10x3xf32>) -> tensor<1x240x1xf32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<1x8x10x3xf32>) -> tensor<1x240x1xf32>
  func.return %0 : tensor<1x240x1xf32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2]

// -----

// CHECK-LABEL: func @reshape1_4D_4D
func.func @reshape1_4D_4D(%arg0: tensor<4x512x1x1xi32>) -> tensor<1x4x1x512xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<4x512x1x1xi32>) -> tensor<1x4x1x512xi32>
  func.return %0 : tensor<1x4x1x512xi32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2, 3]

// -----

// CHECK-LABEL: func @reshape2_4D_4D
func.func @reshape2_4D_4D(%arg0: tensor<4x1x1x1024xi32>) -> tensor<4x1024x1x1xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<4x1x1x1024xi32>) -> tensor<4x1024x1x1xi32>
  func.return %0 : tensor<4x1024x1x1xi32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2, 3]

// -----

// CHECK-LABEL: func @reshape_dynamic_in
func.func @reshape_dynamic_in(%arg0: tensor<?x?xf32>) -> tensor<2x4x5xf32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<?x?xf32>) -> tensor<2x4x5xf32>
  func.return %0 : tensor<2x4x5xf32>
}
// CHECK: %[[FLATTEN:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
// CHECK: %[[CAST:.*]] = tensor.cast %[[FLATTEN]] : tensor<?xf32> to tensor<40xf32>
// CHECK: tensor.expand_shape %[[CAST]] {{\[}}[0, 1, 2]] : tensor<40xf32> into tensor<2x4x5xf32>

// -----

// CHECK-LABEL: func @reshape_1D_2D_dynamic
func.func @reshape_1D_2D_dynamic(%arg0: tensor<?xi32>) -> tensor<1x3xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<?xi32>) -> tensor<1x3xi32>
  func.return %0 : tensor<1x3xi32>
}
// CHECK: %[[CAST:.*]] = tensor.cast %{{.*}} : tensor<?xi32> to tensor<3xi32>
// CHECK: tensor.expand_shape %[[CAST]] {{\[}}[0, 1]] : tensor<3xi32> into tensor<1x3xi32>

// -----

// CHECK-LABEL: func @reshape_2D_1D_dynamic
func.func @reshape_2D_1D_dynamic(%arg0: tensor<?x?xi32>) -> tensor<3xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<?x?xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}
// CHECK: %[[FLATTEN:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0, 1]] : tensor<?x?xi32> into tensor<?xi32>
// CHECK: %[[CAST:.*]] = tensor.cast %[[FLATTEN]] : tensor<?xi32> to tensor<3xi32>
// CHECK: return %[[CAST:.*]] : tensor<3xi32>

// -----
// CHECK-LABEL: func @reshape_2D_1D_semidynamic
func.func @reshape_2D_1D_semidynamic(%arg0: tensor<1x?xi32>) -> tensor<1xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<1x?xi32>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}
// CHECK: %[[CAST:.*]] = tensor.cast %{{.*}} : tensor<1x?xi32> to tensor<1x1xi32>
// CHECK: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[CAST]] {{\[}}[0, 1]] : tensor<1x1xi32> into tensor<1xi32>
// CHECK: return %[[COLLAPSE:.*]] : tensor<1xi32>

// -----

// CHECK-LABEL: func @reshape_1D_0D_dynamic
func.func @reshape_1D_0D_dynamic(%arg0: tensor<?xi32>) -> tensor<i32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<?xi32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK: %[[CAST:.*]] = tensor.cast %{{.*}} : tensor<?xi32> to tensor<1xi32>
// CHECK: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[CAST]] {{\[}}] : tensor<1xi32> into tensor<i32>
// CHECK: return %[[COLLAPSE:.*]] : tensor<i32>

// -----

// CHECK-LABEL: func @reshape_2D_0D_dynamic
func.func @reshape_2D_0D_dynamic(%arg0: tensor<?x?xi32>) -> tensor<i32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<?x?xi32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK: %[[CAST:.*]] = tensor.cast %{{.*}} : tensor<?x?xi32> to tensor<1x1xi32>
// CHECK: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[CAST]] {{\[}}] : tensor<1x1xi32> into tensor<i32>
// CHECK: return %[[COLLAPSE:.*]] : tensor<i32>

// -----

// CHECK-LABEL: func @reshape_3D_1D_semidynamic
func.func @reshape_3D_1D_semidynamic(%arg0: tensor<16x1x?xi32>) -> tensor<16xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<16x1x?xi32>) -> tensor<16xi32>
  func.return %0 : tensor<16xi32>
}
// CHECK: %[[CAST:.*]] = tensor.cast %{{.*}} : tensor<16x1x?xi32> to tensor<16x1x1xi32>
// CHECK: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[CAST]] {{\[}}[0, 1, 2]] : tensor<16x1x1xi32> into tensor<16xi32>
// CHECK: return %[[COLLAPSE:.*]] : tensor<16xi32>

// -----

// CHECK-LABEL: func @reshape_empty
func.func @reshape_empty(%arg0: tensor<7x0xf64>) -> tensor<0x42x101xf64> {
  %0 = stablehlo.reshape %arg0 : (tensor<7x0xf64>) -> tensor<0x42x101xf64>
  return %0 : tensor<0x42x101xf64>
}

// CHECK: %[[INIT:.*]] = tensor.empty
// CHECK: return %[[INIT]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 2)>
// CHECK-DAG: #[[RESULT_MAP:.*]]  = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @reverse
func.func @reverse(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %result = "stablehlo.reverse"(%input) {
    dimensions = dense<1> : tensor<1xi64>, someattr
  } : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %result : tensor<2x3xf32>
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: {someattr}

// -----

// CHECK-LABEL: set_dimension_size
// CHECK-SAME: %[[VALUE:.*]]: tensor<2x?xf32, #stablehlo.bounds<?, 2>
func.func @set_dimension_size(
  %value: tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>,
  %dimension: tensor<i32>)
  -> tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>> {
  // CHECK: tensor.extract_slice %[[VALUE]][0, 0] [2, %{{.*}}] [1, 1] : tensor<2x?xf32, #stablehlo.bounds<?, 2>> to tensor<2x?xf32, #stablehlo.bounds<?, 2>>
  %0 = "stablehlo.set_dimension_size"(%value, %dimension) { dimension = 1 }
    : (tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>, tensor<i32>)
    -> tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>
  func.return %0 : tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>
}

// -----

func.func @torch_index_select(%arg0: tensor<5x1x5xi32>,
                         %arg1: tensor<2xi32>) ->  tensor<2x1x5xi32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
    dim = 0 : i64,
    batch_dims = 0 : i64,
    someattr
  } : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK: func @torch_index_select
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
//      CHECK: %[[INIT1:.+]] = tensor.empty() :
//      CHECK: %[[INIT2:.+]] = tensor.empty() :
//      CHECK: linalg.generic {
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[INDEX]], %[[INIT1]] :
// CHECK-SAME: outs(%[[INIT2]] :
// CHECK-SAME: {someattr}
//      CHECK: ^{{.+}}(%[[VAL:.+]]: i32, %{{.+}}: i32, %{{.+}}: i32):
//      CHECK:   %[[CAST:.+]] = arith.index_cast %[[VAL]] : i32 to index
//      CHECK:   %[[J:.+]] = linalg.index 1
//      CHECK:   %[[K:.+]] = linalg.index 2
//      CHECK:   %[[VAL2:.+]] = tensor.extract %[[INPUT]][%[[CAST]], %[[J]], %[[K]]] : tensor<5x1x5xi32>
//      CHECK:   linalg.yield %[[VAL2]] : i32

// -----

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK: func @torch_index_select_unsigned
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
func.func @torch_index_select_unsigned(%arg0: tensor<5x1x5xui32>,
                                       %arg1: tensor<2xi32>) ->  tensor<2x1x5xui32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
    dim = 0 : i64,
    batch_dims = 0 : i64
  } : (tensor<5x1x5xui32>, tensor<2xi32>) -> tensor<2x1x5xui32>
  func.return %0 : tensor<2x1x5xui32>
}
//      CHECK:   %[[INPUT_SIGNLESS:.*]] = builtin.unrealized_conversion_cast %[[INPUT]] : tensor<5x1x5xui32> to tensor<5x1x5xi32>
//      CHECK:   %[[INIT:.*]] = tensor.empty() : tensor<1x5xi32>
//      CHECK:   %[[RES:.+]] = linalg.generic {
// CHECK-SAME:     indexing_maps
// CHECK-SAME:     #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:   ins(%[[INDEX]], %[[INIT]] : tensor<2xi32>, tensor<1x5xi32>)
//      CHECK:   ^{{.+}}(%[[VAL:.+]]: i32, %{{.+}}: i32, %{{.+}}: i32):
//      CHECK:     %[[CAST:.+]] = arith.index_cast %[[VAL]] : i32 to index
//      CHECK:     %[[J:.+]] = linalg.index 1
//      CHECK:     %[[K:.+]] = linalg.index 2
//      CHECK:     %[[VAL2:.+]] = tensor.extract %[[INPUT_SIGNLESS]][%[[CAST]], %[[J]], %[[K]]] : tensor<5x1x5xi32>
//      CHECK:     linalg.yield %[[VAL2]] : i32
//      CHECK:   %[[RES_UNSIGNED:.+]] = builtin.unrealized_conversion_cast %[[RES]] : tensor<2x1x5xi32> to tensor<2x1x5xui32>
//      CHECK:   return %[[RES_UNSIGNED]]

// -----

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0) -> ()>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0)>
//      CHECK: func @torch_index_select_scalar
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
func.func @torch_index_select_scalar(%arg0: tensor<4x8xf32>,
                                %arg1: tensor<i32>) -> tensor<8xf32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
    batch_dims = 0 : i64,
    dim = 0 : i64
  } : (tensor<4x8xf32>, tensor<i32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}
//      CHECK: %[[T0:.+]] = tensor.empty() : tensor<8xf32>
//      CHECK: %[[T1:.+]] = tensor.empty() : tensor<8xf32>
//      CHECK: linalg.generic {
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP1]]
// CHECK-SAME:   iterator_types = ["parallel"]
// CHECK-SAME:   ins(%[[INDEX]], %[[T0]] : tensor<i32>, tensor<8xf32>) outs(%[[T1]] : tensor<8xf32>)
//      CHECK:   ^{{.+}}(%[[VAL:[a-zA-Z0-9_]+]]: i32, %{{.+}}: f32):
//      CHECK:     %[[CAST:.+]] = arith.index_cast %[[VAL]] : i32 to index
//      CHECK:     %[[I:.+]] = linalg.index 0
//      CHECK:     %[[VAL2:.+]] = tensor.extract %[[INPUT]][%[[CAST]], %[[I]]] : tensor<4x8xf32>
//      CHECK:     linalg.yield %[[VAL2]] : f32

// -----

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func @torch_index_select_batch
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
func.func @torch_index_select_batch(%arg0: tensor<4x7x8x2xf32>,
                               %arg1: tensor<4x1xi32>) -> tensor<4x7x1x2xf32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
    dim = 2 : i64,
    batch_dims = 1 : i64
  } : (tensor<4x7x8x2xf32>, tensor<4x1xi32>) -> tensor<4x7x1x2xf32>
  func.return %0 : tensor<4x7x1x2xf32>
}
//      CHECK: %[[INIT:.+]] = tensor.empty() : tensor<4x7x2xf32>
//      CHECK: linalg.generic {
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[INDEX]], %[[INIT]] :
// CHECK-NEXT: ^{{.+}}(%[[VAL:.+]]: i32, %{{.+}}: f32, %{{.+}}: f32):
//      CHECK:   %[[CAST:.+]] = arith.index_cast %[[VAL]] : i32 to index
//      CHECK:   %[[I:.+]] = linalg.index 0
//      CHECK:   %[[J:.+]] = linalg.index 1
//      CHECK:   %[[L:.+]] = linalg.index 3
//      CHECK:   %[[VAL2:.+]] = tensor.extract %[[INPUT]][%[[I]], %[[J]], %[[CAST]], %[[L]]] : tensor<4x7x8x2xf32>
//      CHECK:   linalg.yield %[[VAL2]] : f32

// -----

//      CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
//      CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//      CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func @torch_index_select_dynamic
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
func.func @torch_index_select_dynamic(%input: tensor<?x?x?x?xf32>,
                                 %index: tensor<?x?xi32>) -> tensor<?x?x?x?xf32>{
  %0 = "stablehlo.torch_index_select"(%input, %index) {
    batch_dims = 1 : i64,
    dim = 2 : i64
  } : (tensor<?x?x?x?xf32>, tensor<?x?xi32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//      CHECK:   %[[D0:.+]] = tensor.dim %[[INPUT]], %[[C0]]
//      CHECK:   %[[D1:.+]] = tensor.dim %[[INPUT]], %[[C1]]
//      CHECK:   %[[D2:.+]] = tensor.dim %[[INDEX]], %[[C1]]
//      CHECK:   %[[D3:.+]] = tensor.dim %[[INPUT]], %[[C3]]
//      CHECK:   %[[D4:.+]] = tensor.dim %[[INPUT]], %[[C0]]
//      CHECK:   %[[D5:.+]] = tensor.dim %[[INPUT]], %[[C1]]
//      CHECK:   %[[D6:.+]] = tensor.dim %[[INPUT]], %[[C3]]
//      CHECK:   %[[INIT0:.+]] = tensor.empty(%[[D4]], %[[D5]], %[[D6]]) : tensor<?x?x?xf32>
//      CHECK:   %[[INIT1:.+]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]], %[[D3]])
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[INDEX]], %[[INIT0]] : tensor<?x?xi32>, tensor<?x?x?xf32>)
// CHECK-SAME:     outs(%[[INIT1]] : tensor<?x?x?x?xf32>)
//      CHECK:     ^{{.+}}(
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9_]+]]: i32, %{{[a-zA-Z0-9_]+}}: f32, %{{[a-zA-Z0-9_]+}}: f32)
//      CHECK:       %[[POS:.+]] = arith.index_cast %[[ARG0]]
//      CHECK:       %[[IDX0:.+]] = linalg.index 0
//      CHECK:       %[[IDX1:.+]] = linalg.index 1
//      CHECK:       %[[IDX3:.+]] = linalg.index 3
//      CHECK:       %[[YIELD:.+]] = tensor.extract %[[INPUT]][%[[IDX0]], %[[IDX1]], %[[POS]], %[[IDX3]]]
//      CHECK:       linalg.yield %[[YIELD]]

// -----

// CHECK-LABEL: func @slice_whole_stride
//       CHECK:   tensor.extract_slice %{{.*}}[1, 0] [1, 4] [1, 1] : tensor<3x4xi32> to tensor<1x4xi32>
func.func @slice_whole_stride(%arg0: tensor<3x4xi32>) -> tensor<1x4xi32> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[2, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: func @slice_stride_part
//       CHECK:   tensor.extract_slice %{{.*}}[1, 1] [1, 2] [1, 1]  : tensor<3x4xi32> to tensor<1x2xi32>
func.func @slice_stride_part(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = dense<[1, 1]> : tensor<2xi64>,
    limit_indices = dense<[2, 3]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @slice_with_strides
//       CHECK:   tensor.extract_slice %{{.*}}[0] [6] [2] : tensor<13xi32> to tensor<6xi32>
func.func @slice_with_strides(%arg0: tensor<13xi32>) -> tensor<6xi32> {
  %0 = "stablehlo.slice"(%arg0) {
    limit_indices = dense<12> : tensor<1xi64>,
    start_indices = dense<0> : tensor<1xi64>,
    strides = dense<2> : tensor<1xi64>
  } : (tensor<13xi32>) -> tensor<6xi32>
  func.return %0 : tensor<6xi32>
}

// -----

// CHECK-LABEL: func @slice_with_strides
//       CHECK:   tensor.extract_slice %{{.*}}[0] [3] [2] : tensor<6xi32> to tensor<3xi32>
func.func @slice_with_strides2(%arg0: tensor<6xi32>) -> tensor<3xi32> {
  %0 = "stablehlo.slice"(%arg0) {
    limit_indices = dense<5> : tensor<1xi64>,
    start_indices = dense<0> : tensor<1xi64>,
    strides = dense<2> : tensor<1xi64>
  } : (tensor<6xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: func @slice_with_empty_result
//       CHECK:   tensor.extract_slice %{{.*}}[0, 2, 0] [3, 0, 5] [1, 2, 1] : tensor<3x3x5xf64> to tensor<3x0x5xf64>
func.func @slice_with_empty_result(%arg0: tensor<3x3x5xf64>) -> tensor<3x0x5xf64> {
  %0 = "stablehlo.slice"(%arg0) {
    limit_indices = dense<[3, 2, 5]> : tensor<3xi64>,
    start_indices = dense<[0, 2, 0]> : tensor<3xi64>,
    strides = dense<[1, 2, 1]> : tensor<3xi64>
  } : (tensor<3x3x5xf64>) -> tensor<3x0x5xf64>
  func.return %0 : tensor<3x0x5xf64>
}

// -----

// CHECK-LABEL: func @dynamic_slice(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
func.func @dynamic_slice(%arg: tensor<3x4xf32>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x4xf32> {
  %0 = "stablehlo.dynamic_slice"(%arg, %start1, %start2) {
    slice_sizes = dense<[1, 4]> : tensor<2xi64>
  } : (tensor<3x4xf32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract %[[ARG1]][] : tensor<i64>
// CHECK:         %[[SCALAR1:.*]] = arith.index_cast %[[EXTRACT1]]
// CHECK:         %[[T1:.*]] = arith.maxsi %[[SCALAR1]], %[[C0]] : index
// CHECK:         %[[CLAMPED1:.*]] = arith.minsi %[[T1]], %[[C2]] : index
// CHECK:         %[[EXTRACT2:.*]] = tensor.extract %[[ARG2]][] : tensor<i64>
// CHECK:         %[[SCALAR2:.*]] = arith.index_cast %[[EXTRACT2]]
// CHECK:         %[[T2:.*]] = arith.maxsi %[[SCALAR2]], %[[C0]] : index
// CHECK:         %[[CLAMPED2:.*]] = arith.minsi %[[T2]], %[[C0]] : index
// CHECK:           tensor.extract_slice %[[ARG0]][%[[CLAMPED1]], %[[CLAMPED2]]] [1, 4] [1, 1]

// -----

// CHECK-LABEL: func @dynamic_slice_unsigned_index(
func.func @dynamic_slice_unsigned_index(
    %arg: tensor<3x4xui32>, %start1: tensor<ui64>, %start2: tensor<ui64>)
    -> tensor<1x4xui32> {
  %0 = "stablehlo.dynamic_slice"(%arg, %start1, %start2) {
    slice_sizes = dense<[1, 4]> : tensor<2xi64>
  } : (tensor<3x4xui32>, tensor<ui64>, tensor<ui64>) -> tensor<1x4xui32>
  func.return %0 : tensor<1x4xui32>
}

// CHECK:         %[[EXTRACT1:.*]] = tensor.extract
// CHECK:         arith.index_castui %[[EXTRACT1]]

// -----

// CHECK-LABEL: func @dynamic_slice_unsigned(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
func.func @dynamic_slice_unsigned(%arg: tensor<3x4xui32>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x4xui32> {
  %0 = "stablehlo.dynamic_slice"(%arg, %start1, %start2) {
    slice_sizes = dense<[1, 4]> : tensor<2xi64>
  } : (tensor<3x4xui32>, tensor<i64>, tensor<i64>) -> tensor<1x4xui32>
  func.return %0 : tensor<1x4xui32>
}

// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[SIGNLESS_ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<3x4xui32> to tensor<3x4xi32>
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract %[[ARG1]][] : tensor<i64>
// CHECK:         %[[SCALAR1:.*]] = arith.index_cast %[[EXTRACT1]]
// CHECK:         %[[T1:.*]] = arith.maxsi %[[SCALAR1]], %[[C0]] : index
// CHECK:         %[[CLAMPED1:.*]] = arith.minsi %[[T1]], %[[C2]] : index
// CHECK:         %[[EXTRACT2:.*]] = tensor.extract %[[ARG2]][] : tensor<i64>
// CHECK:         %[[SCALAR2:.*]] = arith.index_cast %[[EXTRACT2]]
// CHECK:         %[[T2:.*]] = arith.maxsi %[[SCALAR2]], %[[C0]] : index
// CHECK:         %[[CLAMPED2:.*]] = arith.minsi %[[T2]], %[[C0]] : index
// CHECK:           tensor.extract_slice %[[SIGNLESS_ARG0]][%[[CLAMPED1]], %[[CLAMPED2]]] [1, 4] [1, 1]

// -----

// CHECK-LABEL: func @dynamic_update_slice(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
func.func @dynamic_update_slice(%target: tensor<3x3xi32>, %update: tensor<2x2xi32>, %c0: tensor<i32>) -> tensor<3x3xi32> {
  %0 = "stablehlo.dynamic_update_slice"(%target, %update, %c0, %c0)
    : (tensor<3x3xi32>, tensor<2x2xi32>, tensor<i32>, tensor<i32>) -> tensor<3x3xi32>
  func.return %0 : tensor<3x3xi32>
}
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR1:.*]] = arith.index_cast %[[EXTRACT1]]
// CHECK:         %[[T1:.*]] = arith.maxsi %[[SCALAR1]], %[[C0]] : index
// CHECK:         %[[CLAMPED1:.*]] = arith.minsi %[[T1]], %[[C1]] : index
// CHECK:         %[[EXTRACT2:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR2:.*]] = arith.index_cast %[[EXTRACT2]]
// CHECK:         %[[T2:.*]] = arith.maxsi %[[SCALAR2]], %[[C0]] : index
// CHECK:         %[[CLAMPED2:.*]] = arith.minsi %[[T2]], %[[C1]] : index
// CHECK:         %[[RES:.*]] = tensor.insert_slice %[[ARG1]] into %[[ARG0]]
// CHECK-SAME:      [%[[CLAMPED1]], %[[CLAMPED2]]] [2, 2] [1, 1]
// CHECK-SAME:    : tensor<2x2xi32> into tensor<3x3xi32>
// CHECK:         return %[[RES]] : tensor<3x3xi32>

// -----

// CHECK-LABEL: func @dynamic_update_slice_unsigned_index(
func.func @dynamic_update_slice_unsigned_index(
    %target: tensor<3x3xi32>, %update: tensor<2x2xi32>,
    %idx: tensor<ui32>) -> tensor<3x3xi32> {
  %0 = "stablehlo.dynamic_update_slice"(%target, %update, %idx, %idx)
    : (tensor<3x3xi32>, tensor<2x2xi32>, tensor<ui32>, tensor<ui32>) -> tensor<3x3xi32>
  func.return %0 : tensor<3x3xi32>
}

// CHECK:         %[[EXTRACT1:.*]] = tensor.extract
// CHECK:         arith.index_castui %[[EXTRACT1]]

// -----

// CHECK-LABEL: func @dynamic_update_slice_unsigned(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
func.func @dynamic_update_slice_unsigned(%target: tensor<3x3xui32>, %update: tensor<2x2xui32>, %c0: tensor<i32>) -> tensor<3x3xui32> {
  %0 = "stablehlo.dynamic_update_slice"(%target, %update, %c0, %c0)
    : (tensor<3x3xui32>, tensor<2x2xui32>, tensor<i32>, tensor<i32>) -> tensor<3x3xui32>
  func.return %0 : tensor<3x3xui32>
}
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[SIGNLESS_UPDATE:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : tensor<2x2xui32> to tensor<2x2xi32>
// CHECK-DAG:     %[[SIGNLESS_TARGET:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<3x3xui32> to tensor<3x3xi32>
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR1:.*]] = arith.index_cast %[[EXTRACT1]]
// CHECK:         %[[T1:.*]] = arith.maxsi %[[SCALAR1]], %[[C0]] : index
// CHECK:         %[[CLAMPED1:.*]] = arith.minsi %[[T1]], %[[C1]] : index
// CHECK:         %[[EXTRACT2:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR2:.*]] = arith.index_cast %[[EXTRACT2]]
// CHECK:         %[[T2:.*]] = arith.maxsi %[[SCALAR2]], %[[C0]] : index
// CHECK:         %[[CLAMPED2:.*]] = arith.minsi %[[T2]], %[[C1]] : index
// CHECK:         %[[SIGNLESS_RES:.*]] = tensor.insert_slice %[[SIGNLESS_UPDATE]] into %[[SIGNLESS_TARGET]]
// CHECK-SAME:      [%[[CLAMPED1]], %[[CLAMPED2]]] [2, 2] [1, 1]
// CHECK-SAME:    : tensor<2x2xi32> into tensor<3x3xi32>
// CHECK:         %[[RES:.*]] = builtin.unrealized_conversion_cast %[[SIGNLESS_RES]] : tensor<3x3xi32> to tensor<3x3xui32>
// CHECK:         return %[[RES]] : tensor<3x3xui32>

// -----

// CHECK-LABEL: func @dynamic_update_slice_float(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
func.func @dynamic_update_slice_float(%target: tensor<3x3xf32>,
                                 %update: tensor<2x2xf32>,
                                 %c0: tensor<i32>) -> tensor<3x3xf32> {
  %0 = "stablehlo.dynamic_update_slice"(%target, %update, %c0, %c0)
    : (tensor<3x3xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR1:.*]] = arith.index_cast %[[EXTRACT1]]
// CHECK:         %[[T1:.*]] = arith.maxsi %[[SCALAR1]], %[[C0]] : index
// CHECK:         %[[CLAMPED1:.*]] = arith.minsi %[[T1]], %[[C1]] : index
// CHECK:         %[[EXTRACT2:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR2:.*]] = arith.index_cast %[[EXTRACT2]]
// CHECK:         %[[T2:.*]] = arith.maxsi %[[SCALAR2]], %[[C0]] : index
// CHECK:         %[[CLAMPED2:.*]] = arith.minsi %[[T2]], %[[C1]] : index
// CHECK:         %[[RES:.*]] = tensor.insert_slice %[[ARG1]] into %[[ARG0]]
// CHECK-SAME:      [%[[CLAMPED1]], %[[CLAMPED2]]] [2, 2] [1, 1]
// CHECK-SAME:    : tensor<2x2xf32> into tensor<3x3xf32>
// CHECK:         return %[[RES]] : tensor<3x3xf32>

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func @transpose
func.func @transpose(%arg0: tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>}
        : (tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32>
  func.return %0 : tensor<3x2x5x9xi32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]

// CHECK-PRIMITIVE-LABEL: func @transpose
// CHECK-PRIMITIVE: linalg.transpose

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func @transpose_dynamic
func.func @transpose_dynamic(%arg0: tensor<?x?x9x?xi32>) -> tensor<?x?x?x9xi32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>, someattr}
        : (tensor<?x?x9x?xi32>) -> tensor<?x?x?x9xi32>
  func.return %0 : tensor<?x?x?x9xi32>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[D0:.*]] = tensor.dim %arg0, %[[C0]]
// CHECK: %[[D1:.*]] = tensor.dim %arg0, %[[C1]]
// CHECK: %[[D3:.*]] = tensor.dim %arg0, %[[C3]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]], %[[D0]], %[[D3]]) : tensor<?x?x?x9xi32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins(%arg0 : tensor<?x?x9x?xi32>) outs(%[[INIT]] : tensor<?x?x?x9xi32>)
// CHECK-SAME: {someattr}

// CHECK-PRIMITIVE-LABEL: func @transpose_dynamic
// CHECK-PRIMITIVE-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-PRIMITIVE-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-PRIMITIVE-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-PRIMITIVE: %[[D0:.*]] = tensor.dim %arg0, %[[C0]]
// CHECK-PRIMITIVE: %[[D1:.*]] = tensor.dim %arg0, %[[C1]]
// CHECK-PRIMITIVE: %[[D3:.*]] = tensor.dim %arg0, %[[C3]]
// CHECK-PRIMITIVE: %[[INIT:.*]] = tensor.empty(%[[D1]], %[[D0]], %[[D3]]) : tensor<?x?x?x9xi32>
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE-SAME: ins(%arg0 : tensor<?x?x9x?xi32>)
// CHECK-PRIMITIVE-SAME: outs(%[[INIT]] : tensor<?x?x?x9xi32>)
// CHECK-PRIMITIVE-SAME: permutation = [1, 0, 3, 2]
// CHECK-PRIMITIVE-SAME: {someattr}

// -----

func.func @transpose_unsigned(%arg0: tensor<2x2xui32>) -> tensor<2x2xui32> {
  %0 = "stablehlo.transpose"(%arg0) {
    permutation = dense<[1, 0]> : tensor<2xi64>,
    result_layout = dense<[0, 1]> : tensor<2xindex>
  } : (tensor<2x2xui32>) -> tensor<2x2xui32>
  return %0 : tensor<2x2xui32>
}

// Regression test. Just check that unsigned ints lower successfully.
// CHECK-LABEL: func @transpose_unsigned
// CHECK-PRIMITIVE-LABEL: func @transpose_unsigned
