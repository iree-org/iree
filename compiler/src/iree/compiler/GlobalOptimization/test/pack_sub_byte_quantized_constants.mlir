// RUN: iree-opt --split-input-file --iree-global-opt-pack-sub-byte-quantized-constants %s | FileCheck %s --check-prefixes=CHECK,I4
// RUN: iree-opt --split-input-file --iree-global-opt-pack-sub-byte-quantized-constants="min-bit-width=2" %s | FileCheck %s --check-prefixes=CHECK,I2

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Signed 4-bit values are packed two per byte, element 2i in the low nibble of
// byte i, and read back through a bitcast. The pass trusts
// quant_min/quant_max and does not read the data.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @pack_signed_i4(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_signed_i4(
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight
//   CHECK-DAG:   %[[EVEN:.+]] = tensor.extract_slice %[[WEIGHT]][0, 0] [4, 4] [1, 2] : tensor<4x8xi8> to tensor<4x4xi8>
//   CHECK-DAG:   %[[ODD:.+]] = tensor.extract_slice %[[WEIGHT]][0, 1] [4, 4] [1, 2] : tensor<4x8xi8> to tensor<4x4xi8>
//       CHECK:   %[[PACKED:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[EVEN]], %[[ODD]] : tensor<4x4xi8>, tensor<4x4xi8>)
//       CHECK:   ^bb0(%[[LO:.+]]: i8, %[[HI:.+]]: i8, %{{.+}}: i8):
//   CHECK-DAG:     %[[C15:.+]] = arith.constant 15 : i8
//   CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : i8
//   CHECK-DAG:     %[[LO_BITS:.+]] = arith.andi %[[LO]], %[[C15]] : i8
//   CHECK-DAG:     %[[HI_BITS:.+]] = arith.shli %[[HI]], %[[C4]] : i8
//       CHECK:     %[[BYTE:.+]] = arith.ori %[[LO_BITS]], %[[HI_BITS]] : i8
//       CHECK:     linalg.yield %[[BYTE]] : i8
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %[[PACKED]] : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -8, quant_max = 7
//  CHECK-SAME:     ins(%[[NARROW]], %{{.+}} : tensor<4x8xi4>, tensor<4xf32>)

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// PT2E form: a float weight is quantized and dequantized again. Without its
// own range, the dequantize reads the range the quantize clamps to.
util.global private @weight = dense<0.5> : tensor<4x8xf32>
util.func public @pack_range_from_quantize(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xf32>
  %weight_scale = arith.constant dense<0.25> : tensor<4xf32>
  %quantized_init = tensor.empty() : tensor<4x8xi8>
  %quantized = iree_linalg_ext.quantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %weight_scale : tensor<4x8xf32>, tensor<4xf32>)
      outs(%quantized_init : tensor<4x8xi8>) -> tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2]>
      ins(%quantized, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_range_from_quantize(
//       CHECK:   %[[QUANTIZED:.+]] = iree_linalg_ext.quantize_affine
//       CHECK:   tensor.extract_slice %[[QUANTIZED]][0, 0] [4, 4] [1, 2]
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[NARROW]], %{{.+}} : tensor<4x8xi4>, tensor<4xf32>)

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Unsigned 4-bit values are stored minus 8, which makes them signed: one xor
// flips the top bit of both nibbles. The zero point, read from a global, is
// shifted by the same amount, so input - zero_point is unchanged.
util.global private @weight = dense<15> : tensor<4x8xi8>
util.global private @zero_point = dense<[1, 8, 15, 0]> : tensor<4xi8>
util.func public @pack_unsigned_i4_zero_point(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %zero_point = util.global.load @zero_point : tensor<4xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #row2, #id2], quant_min = 0, quant_max = 15,
       input_unsigned, zp_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_unsigned_i4_zero_point(
//       CHECK:   %[[PACKED:.+]] = linalg.generic
//       CHECK:     %[[BYTE:.+]] = arith.ori
//       CHECK:     %[[TOP_BITS:.+]] = arith.constant -120 : i8
//       CHECK:     %[[SHIFTED:.+]] = arith.xori %[[BYTE]], %[[TOP_BITS]] : i8
//       CHECK:     linalg.yield %[[SHIFTED]] : i8
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %[[PACKED]] : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[ZERO_POINT:.+]] = arith.constant dense<[-7, 0, 7, -8]> : tensor<4xi8>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[NARROW]], %{{.+}}, %[[ZERO_POINT]] : tensor<4x8xi4>, tensor<4xf32>, tensor<4xi8>)

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>
#scalar2 = affine_map<(d0, d1) -> ()>

// A scalar zero point from a constant is shifted like a tensor.
util.global private @weight = dense<9> : tensor<4x8xi8>
util.func public @pack_scalar_zero_point(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %zero_point = arith.constant 3 : i8
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #scalar2, #id2], quant_min = 0, quant_max = 15,
       input_unsigned, zp_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, i8)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_scalar_zero_point(
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[ZERO_POINT:.+]] = arith.constant -5 : i8
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[NARROW]], %{{.+}}, %[[ZERO_POINT]] : tensor<4x8xi4>, tensor<4xf32>, i8)

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// A zero point stored as a resource blob: i64 values [1, 8, 15, 0], as PT2E
// emits them. The shifted zero point keeps its element type.
util.global private @weight = dense<3> : tensor<4x8xi8>
util.global private @zero_point = dense_resource<zero_point_blob> : tensor<4xi64>
util.func public @pack_resource_zero_point(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %zero_point = util.global.load @zero_point : tensor<4xi64>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #row2, #id2], quant_min = 0, quant_max = 15,
       input_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi64>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_resource_zero_point(
//       CHECK:   flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   arith.constant dense<[-7, 0, 7, -8]> : tensor<4xi64>

{-#
  dialect_resources: {
    builtin: {
      zero_point_blob: "0x08000000010000000000000008000000000000000F000000000000000000000000000000"
    }
  }
#-}

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The narrow range [-7, 7] also fits i4. An inline constant is packed like a
// global.
util.func public @pack_narrow_range_constant(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = arith.constant dense<-7> : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -7, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_narrow_range_constant(
//       CHECK:   flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Signed 2-bit values are packed into i4 by default. With min-bit-width=2
// they are packed four per byte, element 4i + j in bits [2j, 2j + 2) of byte
// i.
util.global private @weight = dense<-2> : tensor<4x8xi8>
util.func public @pack_signed_i2(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -2, quant_max = 1>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_signed_i2(
//          I4:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//          I4:   iree_linalg_ext.dequantize_affine
//     I4-SAME:     ins(%[[NARROW]], %{{.+}} : tensor<4x8xi4>, tensor<4xf32>)
//          I2:   %[[WEIGHT:.+]] = util.global.load @weight
//      I2-DAG:   %[[LANE0:.+]] = tensor.extract_slice %[[WEIGHT]][0, 0] [4, 2] [1, 4]
//      I2-DAG:   %[[LANE1:.+]] = tensor.extract_slice %[[WEIGHT]][0, 1] [4, 2] [1, 4]
//      I2-DAG:   %[[LANE2:.+]] = tensor.extract_slice %[[WEIGHT]][0, 2] [4, 2] [1, 4]
//      I2-DAG:   %[[LANE3:.+]] = tensor.extract_slice %[[WEIGHT]][0, 3] [4, 2] [1, 4]
//          I2:   %[[PACKED:.+]] = linalg.generic
//     I2-SAME:     ins(%[[LANE0]], %[[LANE1]], %[[LANE2]], %[[LANE3]] :
//          I2:   ^bb0(%[[V0:.+]]: i8, %[[V1:.+]]: i8, %[[V2:.+]]: i8, %[[V3:.+]]: i8, %{{.+}}: i8):
//      I2-DAG:     %[[C3:.+]] = arith.constant 3 : i8
//      I2-DAG:     %[[C2:.+]] = arith.constant 2 : i8
//      I2-DAG:     %[[C4:.+]] = arith.constant 4 : i8
//      I2-DAG:     %[[C6:.+]] = arith.constant 6 : i8
//      I2-DAG:     %[[M0:.+]] = arith.andi %[[V0]], %[[C3]] : i8
//      I2-DAG:     %[[M1:.+]] = arith.andi %[[V1]], %[[C3]] : i8
//      I2-DAG:     %[[M2:.+]] = arith.andi %[[V2]], %[[C3]] : i8
//      I2-DAG:     %[[S1:.+]] = arith.shli %[[M1]], %[[C2]] : i8
//      I2-DAG:     %[[S2:.+]] = arith.shli %[[M2]], %[[C4]] : i8
//      I2-DAG:     %[[S3:.+]] = arith.shli %[[V3]], %[[C6]] : i8
//      I2-DAG:     %[[O1:.+]] = arith.ori %[[M0]], %[[S1]] : i8
//      I2-DAG:     %[[O2:.+]] = arith.ori %[[O1]], %[[S2]] : i8
//      I2-DAG:     %[[O3:.+]] = arith.ori %[[O2]], %[[S3]] : i8
//          I2:     linalg.yield %[[O3]] : i8
//          I2:   %[[NARROW:.+]] = flow.tensor.bitcast %[[PACKED]] : tensor<4x2xi8> -> tensor<4x8xi2>
//          I2:   iree_linalg_ext.dequantize_affine
//     I2-SAME:     ins(%[[NARROW]], %{{.+}} : tensor<4x8xi2>, tensor<4xf32>)

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The unsigned 2-bit range [0, 3] is shifted by the half range of the storage
// it is packed into: by 8 for i4, which leaves a zero point of -6, and by 2 for
// i2, which leaves 0 and drops the zero point.
util.global private @weight = dense<3> : tensor<4x8xi8>
util.func public @pack_unsigned_i2(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %zero_point = arith.constant dense<2> : tensor<4xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #row2, #id2], quant_min = 0, quant_max = 3,
       input_unsigned, zp_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_unsigned_i2(
//          I4:   arith.constant -120 : i8
//          I4:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//          I4:   %[[ZERO_POINT:.+]] = arith.constant dense<-6> : tensor<4xi8>
//          I4:   iree_linalg_ext.dequantize_affine
//     I4-SAME:     quant_min = -8, quant_max = -5>
//     I4-SAME:     ins(%[[NARROW]], %{{.+}}, %[[ZERO_POINT]] : tensor<4x8xi4>, tensor<4xf32>, tensor<4xi8>)
//          I2:   arith.constant -86 : i8
//          I2:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x2xi8> -> tensor<4x8xi2>
//          I2:   iree_linalg_ext.dequantize_affine
//     I2-SAME:     quant_min = -2, quant_max = 1>
//     I2-SAME:     ins(%[[NARROW]], %{{.+}} : tensor<4x8xi2>, tensor<4xf32>)

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Rows of 6 elements do not fill whole bytes at four elements per byte, so
// even with min-bit-width=2 a 2-bit range is packed into i4.
util.global private @weight = dense<1> : tensor<4x6xi8>
util.func public @pack_i2_range_into_i4(%scale: tensor<4xf32>) -> tensor<4x6xf32> {
  %weight = util.global.load @weight : tensor<4x6xi8>
  %init = tensor.empty() : tensor<4x6xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -2, quant_max = 1>
      ins(%weight, %scale : tensor<4x6xi8>, tensor<4xf32>)
      outs(%init : tensor<4x6xf32>) -> tensor<4x6xf32>
  util.return %dequantized : tensor<4x6xf32>
}
// CHECK-LABEL: util.func public @pack_i2_range_into_i4(
//       CHECK:   flow.tensor.bitcast %{{.+}} : tensor<4x3xi8> -> tensor<4x6xi4>

// -----

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#block3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Block-quantized [N, K/B, B] weights, as MatMulNBits imports them, are packed
// along B. Shifting the default zero point 8 to signed storage leaves 0, so the
// zero point is dropped and the dequantize becomes symmetric.
util.global private @weight = dense<9> : tensor<4x2x8xi8>
util.func public @pack_blockwise(%scale: tensor<4x2xf32>) -> tensor<4x2x8xf32> {
  %weight = util.global.load @weight : tensor<4x2x8xi8>
  %zero_point = arith.constant dense<8> : tensor<4x2xi8>
  %init = tensor.empty() : tensor<4x2x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id3, #block3, #block3, #id3], quant_min = 0, quant_max = 15,
       input_unsigned, zp_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x2x8xi8>, tensor<4x2xf32>, tensor<4x2xi8>)
      outs(%init : tensor<4x2x8xf32>) -> tensor<4x2x8xf32>
  util.return %dequantized : tensor<4x2x8xf32>
}
// CHECK-LABEL: util.func public @pack_blockwise(
//       CHECK:   tensor.extract_slice %{{.+}}[0, 0, 0] [4, 2, 4] [1, 1, 2]
//       CHECK:   tensor.extract_slice %{{.+}}[0, 0, 1] [4, 2, 4] [1, 1, 2]
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x2x4xi8> -> tensor<4x2x8xi4>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}], quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[NARROW]], %{{.+}} : tensor<4x2x8xi4>, tensor<4x2xf32>)

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>
#scalar2 = affine_map<(d0, d1) -> ()>

// An unsigned range taken from the quantize: the dequantize has no range of its
// own and gets none, since signed i4 storage implies [-8, 7].
util.global private @weight = dense<0.5> : tensor<4x8xf32>
util.func public @pack_unsigned_range_from_quantize(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xf32>
  %weight_scale = arith.constant dense<0.25> : tensor<4xf32>
  %zero_point = arith.constant 8 : i8
  %quantized_init = tensor.empty() : tensor<4x8xi8>
  %quantized = iree_linalg_ext.quantize_affine
      <indexing_maps = [#id2, #row2, #scalar2, #id2], quant_min = 0, quant_max = 15,
       storage_unsigned, zp_unsigned>
      ins(%weight, %weight_scale, %zero_point : tensor<4x8xf32>, tensor<4xf32>, i8)
      outs(%quantized_init : tensor<4x8xi8>) -> tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #scalar2, #id2], input_unsigned, zp_unsigned>
      ins(%quantized, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, i8)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_unsigned_range_from_quantize(
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     <indexing_maps = [#{{.+}}, #{{.+}}, #{{.+}}]>
//  CHECK-SAME:     ins(%[[NARROW]], %{{.+}} : tensor<4x8xi4>, tensor<4xf32>)

// -----

#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#hwcf_f4 = affine_map<(d0, d1, d2, d3) -> (d3)>
#oihw_o4 = affine_map<(d0, d1, d2, d3) -> (d0)>

// Convolution filters are packed along their innermost dimension: the 16
// output channels of an HWCF filter fill whole bytes, while the 3 columns of
// an OIHW filter do not.
util.global private @hwcf = dense<1> : tensor<3x3x4x16xi8>
util.global private @oihw = dense<1> : tensor<16x4x3x3xi8>
util.func public @pack_hwcf_decline_oihw(%scale: tensor<16xf32>)
    -> (tensor<3x3x4x16xf32>, tensor<16x4x3x3xf32>) {
  %hwcf = util.global.load @hwcf : tensor<3x3x4x16xi8>
  %oihw = util.global.load @oihw : tensor<16x4x3x3xi8>
  %hwcf_init = tensor.empty() : tensor<3x3x4x16xf32>
  %hwcf_dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id4, #hwcf_f4, #id4], quant_min = -8, quant_max = 7>
      ins(%hwcf, %scale : tensor<3x3x4x16xi8>, tensor<16xf32>)
      outs(%hwcf_init : tensor<3x3x4x16xf32>) -> tensor<3x3x4x16xf32>
  %oihw_init = tensor.empty() : tensor<16x4x3x3xf32>
  %oihw_dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id4, #oihw_o4, #id4], quant_min = -8, quant_max = 7>
      ins(%oihw, %scale : tensor<16x4x3x3xi8>, tensor<16xf32>)
      outs(%oihw_init : tensor<16x4x3x3xf32>) -> tensor<16x4x3x3xf32>
  util.return %hwcf_dequantized, %oihw_dequantized
      : tensor<3x3x4x16xf32>, tensor<16x4x3x3xf32>
}
// CHECK-LABEL: util.func public @pack_hwcf_decline_oihw(
//       CHECK:   %[[HWCF:.+]] = flow.tensor.bitcast %{{.+}} : tensor<3x3x4x8xi8> -> tensor<3x3x4x16xi4>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[HWCF]], %{{.+}} : tensor<3x3x4x16xi4>, tensor<16xf32>)
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<16x4x3x3xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// A weight shared by two dequantizes is packed for both.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @pack_shared_input(%scale: tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized0 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  %dequantized1 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized0, %dequantized1 : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_shared_input(
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi4>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi4>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Runtime values are never packed: that would add a dispatch to every call.
util.func public @decline_runtime_value(%input: tensor<4x8xi8>, %scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%input, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_runtime_value(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// A parameter would be packed at program load while its archive still holds
// the unpacked bytes.
util.global private @weight = #flow.parameter.named<"model"::"weight"> : tensor<4x8xi8>
util.func public @decline_parameter(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_parameter(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The second dequantize has no range and keeps reading the i8 weight. Packing
// it for the first one would store the weight twice.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_input_read_unpacked(%scale: tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized0 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  %dequantized1 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2]>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized0, %dequantized1 : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_input_read_unpacked(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Without a range, the values may need all 8 bits.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_no_range(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2]>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_no_range(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// [-8, 8] needs five bits.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_wide_range(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 8>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_wide_range(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The dequantize's own range wins over the narrower range of its quantize.
// [-128, 127] does not fit i4, so the dequantize would fail verification with
// packed storage.
util.global private @weight = dense<0.5> : tensor<4x8xf32>
util.func public @decline_own_range_wider_than_quantize(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xf32>
  %weight_scale = arith.constant dense<0.25> : tensor<4xf32>
  %quantized_init = tensor.empty() : tensor<4x8xi8>
  %quantized = iree_linalg_ext.quantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %weight_scale : tensor<4x8xf32>, tensor<4xf32>)
      outs(%quantized_init : tensor<4x8xi8>) -> tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -128, quant_max = 127>
      ins(%quantized, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_own_range_wider_than_quantize(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// With an odd innermost dimension, row 1 would start in the middle of a byte.
util.global private @weight = dense<1> : tensor<4x7xi8>
util.func public @decline_odd_innermost_dim(%scale: tensor<4xf32>) -> tensor<4x7xf32> {
  %weight = util.global.load @weight : tensor<4x7xi8>
  %init = tensor.empty() : tensor<4x7xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x7xi8>, tensor<4xf32>)
      outs(%init : tensor<4x7xf32>) -> tensor<4x7xf32>
  util.return %dequantized : tensor<4x7xf32>
}
// CHECK-LABEL: util.func public @decline_odd_innermost_dim(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x7xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// [0, 15] read as signed values needs five bits.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_unsigned_range_read_as_signed(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = 0, quant_max = 15>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_unsigned_range_read_as_signed(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// A dynamic innermost dimension cannot be checked for whole bytes per row.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_dynamic_innermost_dim(%scale: tensor<4xf32>) -> tensor<4x?xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %dynamic = tensor.cast %weight : tensor<4x8xi8> to tensor<4x?xi8>
  %c8 = arith.constant 8 : index
  %init = tensor.empty(%c8) : tensor<4x?xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%dynamic, %scale : tensor<4x?xi8>, tensor<4xf32>)
      outs(%init : tensor<4x?xf32>) -> tensor<4x?xf32>
  util.return %dequantized : tensor<4x?xf32>
}
// CHECK-LABEL: util.func public @decline_dynamic_innermost_dim(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x?xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// A tensor without elements has nothing to pack, and its lanes could not be
// sliced.
util.func public @decline_zero_size(%scale: f32) -> tensor<0x16xf32> {
  %weight = arith.constant dense<0> : tensor<0x16xi8>
  %init = tensor.empty() : tensor<0x16xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<0x16xi8>, f32)
      outs(%init : tensor<0x16xf32>) -> tensor<0x16xf32>
  util.return %dequantized : tensor<0x16xf32>
}
// CHECK-LABEL: util.func public @decline_zero_size(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<0x16xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>
#scalar2 = affine_map<(d0, d1) -> ()>

// A zero point wider than 64 bits is out of range for any packed type.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_wide_zero_point(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %zero_point = arith.constant 1267650600228229401496703205376 : i128
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #scalar2, #id2], quant_min = 0, quant_max = 15,
       input_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, i128)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_wide_zero_point(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The zero point 20 is valid for i8 storage but does not fit i4.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.global private @zero_point = dense<[1, 2, 3, 20]> : tensor<4xi8>
util.func public @decline_zero_point_off_grid(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %zero_point = util.global.load @zero_point : tensor<4xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #row2, #id2], quant_min = 0, quant_max = 15,
       input_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_zero_point_off_grid(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// A zero point that is not a constant cannot be checked against i4.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_unknown_zero_point(%scale: tensor<4xf32>, %zero_point: tensor<4xi8>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #row2, #id2], quant_min = 0, quant_max = 15,
       input_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_unknown_zero_point(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Packed storage is signed, and without a zero point there is nothing to
// absorb the shift of unsigned values.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_unsigned_without_zero_point(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = 0, quant_max = 15,
       input_unsigned>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_unsigned_without_zero_point(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// An i2 zero point cannot hold the zero point 1 shifted for i4 storage (-7),
// only the one shifted for i2 storage (-1).
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @narrow_zero_point_type(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %zero_point = arith.constant dense<1> : tensor<4xi2>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #row2, #id2], quant_min = 0, quant_max = 3,
       input_unsigned, zp_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi2>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @narrow_zero_point_type(
//      I4-NOT:   flow.tensor.bitcast
//          I4:   iree_linalg_ext.dequantize_affine
//     I4-SAME:     tensor<4x8xi8>
//          I2:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x2xi8> -> tensor<4x8xi2>
//          I2:   %[[ZERO_POINT:.+]] = arith.constant dense<-1> : tensor<4xi2>
//          I2:   iree_linalg_ext.dequantize_affine
//     I2-SAME:     ins(%[[NARROW]], %{{.+}}, %[[ZERO_POINT]] : tensor<4x8xi2>, tensor<4xf32>, tensor<4xi2>)

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The quantize writes unsigned values that the dequantize reads as signed, so
// the quantize range says nothing about the values the dequantize sees.
util.global private @weight = dense<0.5> : tensor<4x8xf32>
util.func public @decline_quantize_signedness_mismatch(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xf32>
  %weight_scale = arith.constant dense<0.25> : tensor<4xf32>
  %quantized_init = tensor.empty() : tensor<4x8xi8>
  %quantized = iree_linalg_ext.quantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = 0, quant_max = 15,
       storage_unsigned>
      ins(%weight, %weight_scale : tensor<4x8xf32>, tensor<4xf32>)
      outs(%quantized_init : tensor<4x8xi8>) -> tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2]>
      ins(%quantized, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_quantize_signedness_mismatch(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi8>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Only i8 storage is packed: i16 and i4 inputs are left unchanged.
util.global private @wide = dense<1> : tensor<4x8xi16>
util.global private @narrow = dense<1> : tensor<4x8xi4>
util.func public @decline_non_i8_storage(%scale: tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %wide = util.global.load @wide : tensor<4x8xi16>
  %narrow = util.global.load @narrow : tensor<4x8xi4>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized_wide = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%wide, %scale : tensor<4x8xi16>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  %dequantized_narrow = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%narrow, %scale : tensor<4x8xi4>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized_wide, %dequantized_narrow : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_non_i8_storage(
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi16>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     tensor<4x8xi4>
