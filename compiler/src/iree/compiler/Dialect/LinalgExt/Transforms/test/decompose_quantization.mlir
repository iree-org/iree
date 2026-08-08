// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.quantize_affine,iree_linalg_ext.dequantize_affine}))" --split-input-file %s | FileCheck %s

func.func @dequantize_affine_per_channel(%input: tensor<128x64xi8>, %scale: tensor<128xf32>,
    %zp: tensor<128xi8>, %init: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %0 = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%input, %scale, %zp : tensor<128x64xi8>, tensor<128xf32>, tensor<128xi8>)
      outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}
//   CHECK-DAG: #[[$IDENTITY:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[$QPARAM:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @dequantize_affine_per_channel(
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[SCALE:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ZP:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[INIT:[a-zA-Z0-9_]+]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$IDENTITY]], #[[$QPARAM]], #[[$QPARAM]], #[[$IDENTITY]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     ins(%[[INPUT]], %[[SCALE]], %[[ZP]] :
//  CHECK-SAME:     outs(%[[INIT]] :
//       CHECK:   ^bb0(%[[Q:.+]]: i8, %[[S:.+]]: f32, %[[Z:.+]]: i8, %{{.+}}: f32):
// The zero point subtraction widens to i16 first: i8 - i8 needs 9 bits.
//       CHECK:     %[[EXTQ:.+]] = arith.extsi %[[Q]] : i8 to i16
//       CHECK:     %[[EXTZ:.+]] = arith.extsi %[[Z]] : i8 to i16
//       CHECK:     %[[SUB:.+]] = arith.subi %[[EXTQ]], %[[EXTZ]] : i16
//       CHECK:     %[[REAL:.+]] = arith.sitofp %[[SUB]] : i16 to f32
//       CHECK:     %[[MUL:.+]] = arith.mulf %[[REAL]], %[[S]] : f32
//       CHECK:     linalg.yield %[[MUL]]

// -----

// An unsigned storage value zero extends. The subtraction width comes from the
// storage type alone, so a zero point whose element type is wider is narrowed
// to it: its values lie on the storage type's grid whatever type they arrive in.
func.func @dequantize_affine_unsigned_input(%input: tensor<128x64xi8>, %scale: tensor<128xf32>,
    %zp: tensor<128xi32>, %init: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %0 = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>], input_unsigned}
      ins(%input, %scale, %zp : tensor<128x64xi8>, tensor<128xf32>, tensor<128xi32>)
      outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}
// CHECK-LABEL: func.func @dequantize_affine_unsigned_input(
//       CHECK:   ^bb0(%[[Q:.+]]: i8, %[[S:.+]]: f32, %[[Z:.+]]: i32, %{{.+}}: f32):
//       CHECK:     %[[EXTQ:.+]] = arith.extui %[[Q]] : i8 to i16
//       CHECK:     %[[TRUNCZ:.+]] = arith.trunci %[[Z]] : i32 to i16
//       CHECK:     %[[SUB:.+]] = arith.subi %[[EXTQ]], %[[TRUNCZ]] : i16
//       CHECK:     arith.sitofp %[[SUB]] : i16 to f32

// -----

// PT2E emits i64 zero points for i8 data. The subtraction still happens in i16,
// not i64.
func.func @dequantize_affine_i64_zp(%input: tensor<128x64xi8>, %scale: tensor<128xf32>,
    %zp: tensor<128xi64>, %init: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %0 = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = -128 : i64, quant_max = 127 : i64}
      ins(%input, %scale, %zp : tensor<128x64xi8>, tensor<128xf32>, tensor<128xi64>)
      outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}
// CHECK-LABEL: func.func @dequantize_affine_i64_zp(
//       CHECK:   ^bb0(%[[Q:.+]]: i8, %[[S:.+]]: f32, %[[Z:.+]]: i64, %{{.+}}: f32):
//       CHECK:     %[[EXTQ:.+]] = arith.extsi %[[Q]] : i8 to i16
//       CHECK:     %[[TRUNCZ:.+]] = arith.trunci %[[Z]] : i64 to i16
//       CHECK:     %[[SUB:.+]] = arith.subi %[[EXTQ]], %[[TRUNCZ]] : i16
//       CHECK:     arith.sitofp %[[SUB]] : i16 to f32

// -----

// Symmetric dequantization skips the subtraction entirely.
func.func @dequantize_affine_symmetric(%input: tensor<128x64xi4>, %scale: tensor<128xf16>,
    %init: tensor<128x64xf16>) -> tensor<128x64xf16> {
  %0 = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%input, %scale : tensor<128x64xi4>, tensor<128xf16>)
      outs(%init : tensor<128x64xf16>) -> tensor<128x64xf16>
  return %0 : tensor<128x64xf16>
}
// CHECK-LABEL: func.func @dequantize_affine_symmetric(
//       CHECK:   ^bb0(%[[Q:.+]]: i4, %[[S:.+]]: f16, %{{.+}}: f16):
//   CHECK-NOT:     arith.subi
//       CHECK:     %[[REAL:.+]] = arith.sitofp %[[Q]] : i4 to f16
//       CHECK:     %[[MUL:.+]] = arith.mulf %[[REAL]], %[[S]] : f16
//       CHECK:     linalg.yield %[[MUL]]

// -----

// The arithmetic happens in the scale's type, so the i8 goes straight to f32
// and the f32 product is narrowed to the f16 output at the end.
func.func @dequantize_affine_wider_scale(%input: tensor<128x64xi8>,
    %scale: tensor<128xf32>, %init: tensor<128x64xf16>) -> tensor<128x64xf16> {
  %0 = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%input, %scale : tensor<128x64xi8>, tensor<128xf32>)
      outs(%init : tensor<128x64xf16>) -> tensor<128x64xf16>
  return %0 : tensor<128x64xf16>
}
// CHECK-LABEL: func.func @dequantize_affine_wider_scale(
//       CHECK:   ^bb0(%[[Q:.+]]: i8, %[[S:.+]]: f32, %{{.+}}: f16):
//       CHECK:     %[[REAL:.+]] = arith.sitofp %[[Q]] : i8 to f32
//       CHECK:     %[[MUL:.+]] = arith.mulf %[[REAL]], %[[S]] : f32
//       CHECK:     %[[RES:.+]] = arith.truncf %[[MUL]] : f32 to f16
//       CHECK:     linalg.yield %[[RES]]

// -----

func.func @quantize_affine_per_channel(%input: tensor<128x64xf32>, %scale: tensor<128xf32>,
    %zp: tensor<128xi8>, %init: tensor<128x64xi8>) -> tensor<128x64xi8> {
  %0 = iree_linalg_ext.quantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = -128 : i64, quant_max = 127 : i64}
      ins(%input, %scale, %zp : tensor<128x64xf32>, tensor<128xf32>, tensor<128xi8>)
      outs(%init : tensor<128x64xi8>) -> tensor<128x64xi8>
  return %0 : tensor<128x64xi8>
}
// CHECK-LABEL: func.func @quantize_affine_per_channel(
//       CHECK:   ^bb0(%[[X:.+]]: f32, %[[S:.+]]: f32, %[[Z:.+]]: i8, %{{.+}}: i8):
//       CHECK:     %[[DIV:.+]] = arith.divf %[[X]], %[[S]] : f32
//       CHECK:     %[[RND:.+]] = math.roundeven %[[DIV]] : f32
//       CHECK:     %[[ZF:.+]] = arith.sitofp %[[Z]] : i8 to f32
//       CHECK:     %[[ADD:.+]] = arith.addf %[[RND]], %[[ZF]] : f32
//       CHECK:     %[[MIN:.+]] = arith.constant -1.280000e+02 : f32
//       CHECK:     %[[MAX:.+]] = arith.constant 1.270000e+02 : f32
//       CHECK:     %[[LO:.+]] = arith.maxnumf %[[ADD]], %[[MIN]] : f32
//       CHECK:     %[[HI:.+]] = arith.minnumf %[[LO]], %[[MAX]] : f32
//       CHECK:     %[[RES:.+]] = arith.fptosi %[[HI]] : f32 to i8
//       CHECK:     linalg.yield %[[RES]]

// -----

// Narrow range symmetric weights clamp to the declared range, not to the range
// of the storage type, and there is no zero point to add.
func.func @quantize_affine_symmetric_narrow_range(%input: tensor<128x64xf32>,
    %scale: tensor<128xf32>, %init: tensor<128x64xi8>) -> tensor<128x64xi8> {
  %0 = iree_linalg_ext.quantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = -127 : i64, quant_max = 127 : i64}
      ins(%input, %scale : tensor<128x64xf32>, tensor<128xf32>)
      outs(%init : tensor<128x64xi8>) -> tensor<128x64xi8>
  return %0 : tensor<128x64xi8>
}
// CHECK-LABEL: func.func @quantize_affine_symmetric_narrow_range(
//       CHECK:   ^bb0(%[[X:.+]]: f32, %[[S:.+]]: f32, %{{.+}}: i8):
//       CHECK:     %[[DIV:.+]] = arith.divf %[[X]], %[[S]] : f32
//       CHECK:     %[[RND:.+]] = math.roundeven %[[DIV]] : f32
//   CHECK-NOT:     arith.addf
//       CHECK:     %[[MIN:.+]] = arith.constant -1.270000e+02 : f32
//       CHECK:     %[[MAX:.+]] = arith.constant 1.270000e+02 : f32
//       CHECK:     %[[LO:.+]] = arith.maxnumf %[[RND]], %[[MIN]] : f32
//       CHECK:     %[[HI:.+]] = arith.minnumf %[[LO]], %[[MAX]] : f32
//       CHECK:     arith.fptosi %[[HI]] : f32 to i8

// -----

// Unsigned storage converts with fptoui and clamps to the unsigned range.
func.func @quantize_affine_unsigned_storage(%input: tensor<128x64xf32>,
    %scale: tensor<f32>, %zp: tensor<i8>, %init: tensor<128x64xi8>) -> tensor<128x64xi8> {
  %0 = iree_linalg_ext.quantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = 0 : i64, quant_max = 255 : i64,
       storage_unsigned, zp_unsigned}
      ins(%input, %scale, %zp : tensor<128x64xf32>, tensor<f32>, tensor<i8>)
      outs(%init : tensor<128x64xi8>) -> tensor<128x64xi8>
  return %0 : tensor<128x64xi8>
}
// CHECK-LABEL: func.func @quantize_affine_unsigned_storage(
//       CHECK:   ^bb0(%[[X:.+]]: f32, %[[S:.+]]: f32, %[[Z:.+]]: i8, %{{.+}}: i8):
//       CHECK:     arith.uitofp %[[Z]] : i8 to f32
//       CHECK:     %[[MAX:.+]] = arith.constant 2.550000e+02 : f32
//       CHECK:     arith.minnumf %{{.+}}, %[[MAX]] : f32
//       CHECK:     arith.fptoui %{{.+}} : f32 to i8

// -----

// A zero point narrower than the storage type is extended to the same width the
// storage type needs.
func.func @dequantize_affine_narrow_zp(%input: tensor<128x64xi16>, %scale: tensor<128xf32>,
    %zp: tensor<128xi8>, %init: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %0 = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%input, %scale, %zp : tensor<128x64xi16>, tensor<128xf32>, tensor<128xi8>)
      outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}
// CHECK-LABEL: func.func @dequantize_affine_narrow_zp(
//       CHECK:   ^bb0(%[[Q:.+]]: i16, %[[S:.+]]: f32, %[[Z:.+]]: i8, %{{.+}}: f32):
//       CHECK:     %[[EXTQ:.+]] = arith.extsi %[[Q]] : i16 to i32
//       CHECK:     %[[EXTZ:.+]] = arith.extsi %[[Z]] : i8 to i32
//       CHECK:     %[[SUB:.+]] = arith.subi %[[EXTQ]], %[[EXTZ]] : i32
//       CHECK:     arith.sitofp %[[SUB]] : i32 to f32

// -----

// A transpose folded into the op carries straight over to the generic's
// indexing maps; no transpose is materialized.
func.func @dequantize_affine_transposed(%input: tensor<128x64xi8>,
    %scale: tensor<128xf32>, %init: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d1, d0)>]}
      ins(%input, %scale : tensor<128x64xi8>, tensor<128xf32>)
      outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
//   CHECK-DAG: #[[$INPUT:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[$QPARAM:.+]] = affine_map<(d0, d1) -> (d0)>
//   CHECK-DAG: #[[$OUTPUT:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: func.func @dequantize_affine_transposed(
//   CHECK-NOT:   linalg.transpose
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$INPUT]], #[[$QPARAM]], #[[$OUTPUT]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//       CHECK:   ^bb0(%[[Q:.+]]: i8, %[[S:.+]]: f32, %{{.+}}: f32):
//       CHECK:     %[[F:.+]] = arith.sitofp %[[Q]] : i8 to f32
//       CHECK:     arith.mulf %[[F]], %[[S]] : f32

// -----

// The f16 input is extended to the f32 scale before the divide, and the clamp
// bounds are f32 constants rather than f16 ones.
func.func @quantize_affine_wider_scale(%input: tensor<128x64xf16>,
    %scale: tensor<128xf32>, %zp: tensor<128xi8>,
    %init: tensor<128x64xi8>) -> tensor<128x64xi8> {
  %0 = iree_linalg_ext.quantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = -128 : i64, quant_max = 127 : i64}
      ins(%input, %scale, %zp : tensor<128x64xf16>, tensor<128xf32>, tensor<128xi8>)
      outs(%init : tensor<128x64xi8>) -> tensor<128x64xi8>
  return %0 : tensor<128x64xi8>
}
// CHECK-LABEL: func.func @quantize_affine_wider_scale(
//       CHECK:   ^bb0(%[[X:.+]]: f16, %[[S:.+]]: f32, %[[Z:.+]]: i8, %{{.+}}: i8):
//       CHECK:     %[[XF:.+]] = arith.extf %[[X]] : f16 to f32
//       CHECK:     %[[DIV:.+]] = arith.divf %[[XF]], %[[S]] : f32
//       CHECK:     %[[RND:.+]] = math.roundeven %[[DIV]] : f32
//       CHECK:     %[[ZF:.+]] = arith.sitofp %[[Z]] : i8 to f32
//       CHECK:     %[[ADD:.+]] = arith.addf %[[RND]], %[[ZF]] : f32
//       CHECK:     %[[MIN:.+]] = arith.constant -1.280000e+02 : f32
//       CHECK:     %[[MAX:.+]] = arith.constant 1.270000e+02 : f32
//       CHECK:     %[[LO:.+]] = arith.maxnumf %[[ADD]], %[[MIN]] : f32
//       CHECK:     %[[HI:.+]] = arith.minnumf %[[LO]], %[[MAX]] : f32
//       CHECK:     %[[RES:.+]] = arith.fptosi %[[HI]] : f32 to i8
//       CHECK:     linalg.yield %[[RES]]
