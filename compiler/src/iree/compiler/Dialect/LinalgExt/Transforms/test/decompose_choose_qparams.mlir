// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.choose_qparams_affine}))" --split-input-file %s | FileCheck %s

// Each block is reduced to its range, which is then mapped to the parameters
// in the input type. An empty block, whose minimum exceeds its maximum, gets a
// unit scale and a zero zero point.
func.func @choose_qparams_affine_asymmetric(%input: tensor<4x8xf32>,
    %scale: tensor<f64>, %zp: tensor<i64>) -> (tensor<f64>, tensor<i64>) {
  %0:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>],
       mapping = asymmetric, quant_min = -128, quant_max = 127, eps = 1.0e-08>
      ins(%input : tensor<4x8xf32>)
      outs(%scale, %zp : tensor<f64>, tensor<i64>) -> tensor<f64>, tensor<i64>
  return %0#0, %0#1 : tensor<f64>, tensor<i64>
}
//   CHECK-DAG: #[[$IDENTITY:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[$SCALAR:.+]] = affine_map<(d0, d1) -> ()>
//   CHECK-DAG: #[[$PARAMS:.+]] = affine_map<() -> ()>
// CHECK-LABEL: func.func @choose_qparams_affine_asymmetric(
//       CHECK:   %[[POS_INF:.+]] = arith.constant 0x7F800000 : f32
//       CHECK:   %[[MIN_INIT:.+]] = linalg.fill ins(%[[POS_INF]] : f32)
//       CHECK:   %[[NEG_INF:.+]] = arith.constant 0xFF800000 : f32
//       CHECK:   %[[MAX_INIT:.+]] = linalg.fill ins(%[[NEG_INF]] : f32)
//       CHECK:   %[[RANGE:.+]]:2 = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$IDENTITY]], #[[$SCALAR]], #[[$SCALAR]]]
//  CHECK-SAME:     iterator_types = ["reduction", "reduction"]
//  CHECK-SAME:     outs(%[[MIN_INIT]], %[[MAX_INIT]] : tensor<f32>, tensor<f32>)
//       CHECK:     arith.minimumf
//       CHECK:     arith.maximumf
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$PARAMS]], #[[$PARAMS]], #[[$PARAMS]], #[[$PARAMS]]]
//  CHECK-SAME:     ins(%[[RANGE]]#0, %[[RANGE]]#1 : tensor<f32>, tensor<f32>)
//       CHECK:   ^bb0(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %{{.+}}: f64, %{{.+}}: i64):
//       CHECK:     %[[MIN_NEG:.+]] = arith.minimumf %[[MIN]], %[[ZERO:.+]] : f32
//       CHECK:     %[[MAX_POS:.+]] = arith.maximumf %[[MAX]], %[[ZERO]] : f32
//       CHECK:     %[[SPAN:.+]] = arith.subf %[[MAX_POS]], %[[MIN_NEG]] : f32
//       CHECK:     %[[DIV:.+]] = arith.divf %[[SPAN]], %{{.+}} : f32
//       CHECK:     %[[SCALE:.+]] = arith.maximumf %[[DIV]], %{{.+}} : f32
//       CHECK:     %[[DESCALED:.+]] = arith.divf %[[MIN_NEG]], %[[SCALE]] : f32
//       CHECK:     %[[ROUNDED:.+]] = math.roundeven %[[DESCALED]] : f32
//       CHECK:     %[[ZP:.+]] = arith.subf %{{.+}}, %[[ROUNDED]] : f32
//       CHECK:     %[[LO:.+]] = arith.maxnumf %[[ZP]], %{{.+}} : f32
//       CHECK:     %[[HI:.+]] = arith.minnumf %[[LO]], %{{.+}} : f32
//       CHECK:     %[[EMPTY:.+]] = arith.cmpf ogt, %[[MIN]], %[[MAX]] : f32
//       CHECK:     %[[SEL_SCALE:.+]] = arith.select %[[EMPTY]], %{{.+}}, %[[SCALE]] : f32
//       CHECK:     %[[EXT:.+]] = arith.extf %[[SEL_SCALE]] : f32 to f64
//       CHECK:     %[[SEL_ZP:.+]] = arith.select %[[EMPTY]], %[[ZERO]], %[[HI]] : f32
//       CHECK:     %[[INT:.+]] = arith.fptosi %[[SEL_ZP]] : f32 to i64
//       CHECK:     linalg.yield %[[EXT]], %[[INT]] : f64, i64

// -----

func.func @choose_qparams_affine_per_token_min_error(%input: tensor<2x?x8xf32>,
    %scale: tensor<2x?xf32>, %zp: tensor<2x?xi8>) -> (tensor<2x?xf32>, tensor<2x?xi8>) {
  %0:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>],
       mapping = asymmetric_min_error, quant_min = -128, quant_max = 127,
       eps = 1.1920928955078125e-07>
      ins(%input : tensor<2x?x8xf32>)
      outs(%scale, %zp : tensor<2x?xf32>, tensor<2x?xi8>) -> tensor<2x?xf32>, tensor<2x?xi8>
  return %0#0, %0#1 : tensor<2x?xf32>, tensor<2x?xi8>
}
//   CHECK-DAG: #[[$IDENTITY:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//   CHECK-DAG: #[[$TOKEN:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @choose_qparams_affine_per_token_min_error(
//       CHECK:   %[[DIM:.+]] = tensor.dim
//       CHECK:   tensor.empty(%[[DIM]]) : tensor<2x?xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$IDENTITY]], #[[$TOKEN]], #[[$TOKEN]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//       CHECK:   linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//       CHECK:     %[[MIN_ERR:.+]] = arith.addf
//       CHECK:     %[[MAX_ERR:.+]] = arith.addf
//       CHECK:     %[[ERR:.+]] = arith.addf %[[MIN_ERR]], %[[MAX_ERR]] : f32
//       CHECK:     %[[USE_MIN:.+]] = arith.cmpf ogt, %[[ERR]]
//       CHECK:     %[[SEL:.+]] = arith.select %[[USE_MIN]]
//       CHECK:     %[[LO:.+]] = arith.maxnumf %[[SEL]]
//       CHECK:     %[[HI:.+]] = arith.minnumf %[[LO]]
//       CHECK:     math.roundeven %[[HI]] : f32
//       CHECK:     arith.fptosi %{{.+}} : f32 to i8

// -----

// symmetric_qmax clamps the largest magnitude to eps before dividing by
// quant_max. Types without infinities start the reduction from their largest
// finite values.
func.func @choose_qparams_affine_symmetric_qmax(%input: tensor<4x8xf8E4M3FN>,
    %scale: tensor<4xf32>) -> tensor<4xf32> {
  %0 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>],
       mapping = symmetric_qmax, quant_min = -128, quant_max = 127, eps = 1.0e-02>
      ins(%input : tensor<4x8xf8E4M3FN>)
      outs(%scale : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @choose_qparams_affine_symmetric_qmax(
//   CHECK-DAG:   arith.constant 4.480000e+02 : f8E4M3FN
//   CHECK-DAG:   arith.constant -4.480000e+02 : f8E4M3FN
//       CHECK:   linalg.generic
//       CHECK:   linalg.generic
//       CHECK:     %[[NEG:.+]] = arith.negf
//       CHECK:     %[[AMAX:.+]] = arith.maximumf %[[NEG]]
//       CHECK:     %[[CLAMPED:.+]] = arith.maximumf %[[AMAX]]
//       CHECK:     %[[SCALE:.+]] = arith.divf %[[CLAMPED]]
//       CHECK:     arith.select %{{.+}}, %{{.+}}, %[[SCALE]]
//       CHECK:     arith.extf %{{.+}} : f8E4M3FN to f32
//   CHECK-NOT:     arith.fptosi

// -----

// The symmetric zero point is the middle of the range.
func.func @choose_qparams_affine_symmetric_unsigned(%input: tensor<4x8xf32>,
    %scale: tensor<f32>, %zp: tensor<i8>) -> (tensor<f32>, tensor<i8>) {
  %0:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>],
       mapping = symmetric, quant_min = 0, quant_max = 255, eps = 1.0e-08,
       zp_unsigned>
      ins(%input : tensor<4x8xf32>)
      outs(%scale, %zp : tensor<f32>, tensor<i8>) -> tensor<f32>, tensor<i8>
  return %0#0, %0#1 : tensor<f32>, tensor<i8>
}
// CHECK-LABEL: func.func @choose_qparams_affine_symmetric_unsigned(
//       CHECK:   linalg.generic
//       CHECK:   linalg.generic
//       CHECK:     arith.divf %{{.+}}, %[[HALF_RANGE:.+]] : f32
//       CHECK:     %[[ZP:.+]] = arith.constant 1.280000e+02 : f32
//       CHECK:     %[[SEL:.+]] = arith.select %{{.+}}, %{{.+}}, %[[ZP]] : f32
//       CHECK:     arith.fptoui %[[SEL]] : f32 to i8

// -----

// The signed symmetric zero point is 0.
func.func @choose_qparams_affine_symmetric_signed(%input: tensor<4x8xf32>,
    %scale: tensor<f32>, %zp: tensor<i8>) -> (tensor<f32>, tensor<i8>) {
  %0:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>],
       mapping = symmetric, quant_min = -128, quant_max = 127, eps = 1.0e-08>
      ins(%input : tensor<4x8xf32>)
      outs(%scale, %zp : tensor<f32>, tensor<i8>) -> tensor<f32>, tensor<i8>
  return %0#0, %0#1 : tensor<f32>, tensor<i8>
}
// CHECK-LABEL: func.func @choose_qparams_affine_symmetric_signed(
//       CHECK:   linalg.generic
//       CHECK:   linalg.generic
//       CHECK:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:     %[[HALF_RANGE:.+]] = arith.constant 1.275000e+02 : f32
//       CHECK:     arith.divf %{{.+}}, %[[HALF_RANGE]] : f32
//       CHECK:     %[[ZP:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:     %[[SEL:.+]] = arith.select %{{.+}}, %[[ZERO]], %[[ZP]] : f32
//       CHECK:     arith.fptosi %[[SEL]] : f32 to i8

// -----

// A transposed input map reduces the dimension that the scale map does not
// index, here the leading one.
func.func @choose_qparams_affine_transposed_input(%input: tensor<8x4xf32>,
    %scale: tensor<4xf32>) -> tensor<4xf32> {
  %0 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                        affine_map<(d0, d1) -> (d0)>],
       mapping = symmetric_qmax, quant_min = -128, quant_max = 127, eps = 1.0e-05>
      ins(%input : tensor<8x4xf32>)
      outs(%scale : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
//   CHECK-DAG: #[[$TRANSPOSED:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//   CHECK-DAG: #[[$ROW:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @choose_qparams_affine_transposed_input(
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$TRANSPOSED]], #[[$ROW]], #[[$ROW]]]
//  CHECK-SAME:     iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:     ins(%{{.+}} : tensor<8x4xf32>)
//       CHECK:     } -> (tensor<4xf32>, tensor<4xf32>)

// -----

// The arithmetic happens in bf16 and the results are then converted to the
// scale and zero point types.
func.func @choose_qparams_affine_bf16(%input: tensor<4x8xbf16>,
    %scale: tensor<4xf32>, %zp: tensor<4xi8>) -> (tensor<4xf32>, tensor<4xi8>) {
  %0:2 = iree_linalg_ext.choose_qparams_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>],
       mapping = asymmetric, quant_min = -128, quant_max = 127, eps = 1.0e-05>
      ins(%input : tensor<4x8xbf16>)
      outs(%scale, %zp : tensor<4xf32>, tensor<4xi8>) -> tensor<4xf32>, tensor<4xi8>
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xi8>
}
// CHECK-LABEL: func.func @choose_qparams_affine_bf16(
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}}, %{{.+}} : tensor<4xbf16>, tensor<4xbf16>)
//       CHECK:   linalg.generic
//       CHECK:     ^bb0(%{{.+}}: bf16, %{{.+}}: bf16, %{{.+}}: f32, %{{.+}}: i8):
//       CHECK:     %[[DIV:.+]] = arith.divf %{{.+}}, %{{.+}} : bf16
//       CHECK:     %[[SCALE:.+]] = arith.maximumf %[[DIV]], %{{.+}} : bf16
//       CHECK:     arith.divf %{{.+}}, %[[SCALE]] : bf16
//       CHECK:     arith.extf %{{.+}} : bf16 to f32
//       CHECK:     arith.fptosi %{{.+}} : bf16 to i8
