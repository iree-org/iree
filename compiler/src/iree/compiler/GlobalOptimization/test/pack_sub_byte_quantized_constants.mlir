// RUN: iree-opt --split-input-file --iree-global-opt-pack-sub-byte-quantized-constants %s | FileCheck %s

// Which dequantize_affine inputs the pass packs into i2 or i4 storage, and the
// packing IR.

//===----------------------------------------------------------------------===//
// Packing layout
//
// With w-bit storage, k = 8 / w values share a byte: element k * i + j of a
// row goes to bits [j * w, (j + 1) * w) of byte i, the layout narrow-type
// emulation expects. Lane j holds elements j, j + k, ... of each row; it is
// masked to w bits, shifted left by j * w, and ORed with the other lanes. The
// last lane needs no mask, since its shift drops the higher bits. Later cases
// only check the bitcast and the dequantize reading it.
//===----------------------------------------------------------------------===//

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Signed 2-bit values are packed four per byte, and only the three lower lanes
// are masked.
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
//   CHECK-DAG: #[[$ID2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[$ROW2:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: util.func public @pack_signed_i2(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//       CHECK:   %[[LANE0:.+]] = tensor.extract_slice %[[WEIGHT]][0, 0] [4, 2] [1, 4] : tensor<4x8xi8> to tensor<4x2xi8>
//       CHECK:   %[[LANE1:.+]] = tensor.extract_slice %[[WEIGHT]][0, 1] [4, 2] [1, 4] : tensor<4x8xi8> to tensor<4x2xi8>
//       CHECK:   %[[LANE2:.+]] = tensor.extract_slice %[[WEIGHT]][0, 2] [4, 2] [1, 4] : tensor<4x8xi8> to tensor<4x2xi8>
//       CHECK:   %[[LANE3:.+]] = tensor.extract_slice %[[WEIGHT]][0, 3] [4, 2] [1, 4] : tensor<4x8xi8> to tensor<4x2xi8>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x2xi8>
//       CHECK:   %[[PACKED:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$ID2]], #[[$ID2]], #[[$ID2]], #[[$ID2]], #[[$ID2]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     ins(%[[LANE0]], %[[LANE1]], %[[LANE2]], %[[LANE3]] : tensor<4x2xi8>, tensor<4x2xi8>, tensor<4x2xi8>, tensor<4x2xi8>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<4x2xi8>)
//  CHECK-NEXT:   ^bb0(%[[V0:[a-zA-Z0-9_]+]]: i8, %[[V1:[a-zA-Z0-9_]+]]: i8, %[[V2:[a-zA-Z0-9_]+]]: i8, %[[V3:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i8):
// byte = (v0 & 3) | ((v1 & 3) << 2) | ((v2 & 3) << 4) | (v3 << 6)
//   CHECK-DAG:     %[[MASK:.+]] = arith.constant 3 : i8
//   CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : i8
//   CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : i8
//   CHECK-DAG:     %[[C6:.+]] = arith.constant 6 : i8
//   CHECK-DAG:     %[[M0:.+]] = arith.andi %[[V0]], %[[MASK]] : i8
//   CHECK-DAG:     %[[M1:.+]] = arith.andi %[[V1]], %[[MASK]] : i8
//   CHECK-DAG:     %[[M2:.+]] = arith.andi %[[V2]], %[[MASK]] : i8
//   CHECK-DAG:     %[[S1:.+]] = arith.shli %[[M1]], %[[C2]] : i8
//   CHECK-DAG:     %[[S2:.+]] = arith.shli %[[M2]], %[[C4]] : i8
//   CHECK-DAG:     %[[S3:.+]] = arith.shli %[[V3]], %[[C6]] : i8
//   CHECK-DAG:     %[[O1:.+]] = arith.ori %[[M0]], %[[S1]] : i8
//   CHECK-DAG:     %[[O2:.+]] = arith.ori %[[O1]], %[[S2]] : i8
//       CHECK:     %[[BYTE:.+]] = arith.ori %[[O2]], %[[S3]] : i8
//  CHECK-NEXT:     linalg.yield %[[BYTE]] : i8
//       CHECK:   } -> tensor<4x2xi8>
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %[[PACKED]] : tensor<4x2xi8> -> tensor<4x8xi2>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     <indexing_maps = [#[[$ID2]], #[[$ROW2]], #[[$ID2]]], quant_min = -2, quant_max = 1>
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]] : tensor<4x8xi2>, tensor<4xf32>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x8xf32>) -> tensor<4x8xf32>
//       CHECK:   util.return %[[DEQ]] : tensor<4x8xf32>

//===----------------------------------------------------------------------===//
// Ranges
//
// The pass picks the narrowest storage that holds quant_min/quant_max:
// [-2, 1] or [0, 3] for i2, [-8, 7] or [0, 15] for i4, depending on
// input_unsigned.
//===----------------------------------------------------------------------===//

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The lower bound -9 is below the i4 minimum of -8.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_wide_range(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -9, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_wide_range(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -9, quant_max = 7>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Without input_unsigned, [0, 15] is a signed range and needs five bits.
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
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 15>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

//===----------------------------------------------------------------------===//
// Zero points
//
// dequantize_affine requires the zero point to fit the input element type, so
// every zero point value must fit the narrow type. The pass therefore needs a
// constant zero point: a scalar or a tensor, inline, in an immutable global or
// in a resource blob, possibly reshaped. The zero point operand is not changed.
//===----------------------------------------------------------------------===//

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// PT2E per-tensor quantization without folding the quantize: the weight is
// computed from a float global with a scalar scale and zero point.
util.global private @weight = dense<0.5> : tensor<4x8xf32>
util.func public @pack_scalar_zero_point() -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xf32>
  %scale = arith.constant 0.25 : f32
  %zero_point = arith.constant 8 : i64
  %quantized_init = tensor.empty() : tensor<4x8xi8>
  %quantized = iree_linalg_ext.quantize_affine
      <indexing_maps = [#id2, #scalar2, #scalar2, #id2], quant_min = 0, quant_max = 15,
       storage_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xf32>, f32, i64)
      outs(%quantized_init : tensor<4x8xi8>) -> tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #scalar2, #id2], quant_min = 0, quant_max = 15,
       input_unsigned>
      ins(%quantized, %scale, %zero_point : tensor<4x8xi8>, f32, i64)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_scalar_zero_point(
//       CHECK:   %[[SCALE:.+]] = arith.constant 2.500000e-01 : f32
//       CHECK:   %[[ZERO_POINT:.+]] = arith.constant 8 : i64
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]], %[[ZERO_POINT]] : tensor<4x8xi4>, f32, i64)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Unsigned 4-bit weights with a per-channel zero point loaded from a global as
// [4, 1] and collapsed to [4], as the Torch input conversion emits it.
util.global private @weight = dense<15> : tensor<4x8xi8>
util.global private @zero_point = dense<[[1], [8], [15], [0]]> : tensor<4x1xi8>
util.func public @pack_unsigned_i4_zero_point(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %loaded = util.global.load @zero_point : tensor<4x1xi8>
  %zero_point = tensor.collapse_shape %loaded [[0, 1]] : tensor<4x1xi8> into tensor<4xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #row2, #id2], quant_min = 0, quant_max = 15,
       input_unsigned, zp_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_unsigned_i4_zero_point(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]], %{{.+}} : tensor<4x8xi4>, tensor<4xf32>, tensor<4xi8>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// A zero point in a resource blob with i64 elements, [1, 8, 15, 0], which fit
// i4.
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
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 15, input_unsigned>
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]], %{{.+}} : tensor<4x8xi4>, tensor<4xf32>, tensor<4xi64>)
//       CHECK:   util.return %[[DEQ]]

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

// The range [0, 3] fits i2, but its zero point 5 does not, so the weight is
// packed into i4.
util.global private @weight = dense<3> : tensor<4x8xi8>
util.func public @pack_i2_range_into_i4_for_zero_point(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %zero_point = arith.constant dense<5> : tensor<4xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #row2, #id2], quant_min = 0, quant_max = 3,
       input_unsigned, zp_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_i2_range_into_i4_for_zero_point(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 3, input_unsigned, zp_unsigned>
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]], %{{.+}} : tensor<4x8xi4>, tensor<4xf32>, tensor<4xi8>)
//       CHECK:   util.return %[[DEQ]]

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
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[ZERO_POINT:.+]] = util.global.load @zero_point : tensor<4xi8>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = 0, quant_max = 15, input_unsigned>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]], %[[ZERO_POINT]] : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi8>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// A zero point unpacked from packed bytes, as for MatMulNBits, is computed and
// cannot be checked against i4 before constant evaluation.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.global private @packed_zero_point = dense<[1, 18, 3, 4]> : tensor<4xi8>
util.func public @decline_computed_zero_point(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %packed = util.global.load @packed_zero_point : tensor<4xi8>
  %mask = arith.constant dense<15> : tensor<4xi8>
  %zero_point = arith.andi %packed, %mask : tensor<4xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #row2, #id2], quant_min = 0, quant_max = 15,
       input_unsigned>
      ins(%weight, %scale, %zero_point : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_computed_zero_point(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[ZERO_POINT:.+]] = arith.andi
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]], %[[ZERO_POINT]] : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi8>)
//       CHECK:   util.return %[[DEQ]]
//===----------------------------------------------------------------------===//
// Shapes
//
// Packing works per row, while narrow-type emulation indexes the flattened
// tensor. Both agree only when each row starts on a byte boundary, so the
// innermost dimension must be a multiple of 8 / w. The shape must be static
// and non-empty.
//===----------------------------------------------------------------------===//

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Rows of 6 elements do not fill whole bytes at four elements per byte, but
// they do at two, so a 2-bit range is packed into i4.
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
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x3xi8> -> tensor<4x6xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -2, quant_max = 1>
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]] : tensor<4x6xi4>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// A dynamic shape, as of a runtime int4 KV cache, is not packed.
util.func public @decline_dynamic_shape(%input: tensor<?x8xi8>, %scale: tensor<?xf32>) -> tensor<?x8xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %input, %c0 : tensor<?x8xi8>
  %init = tensor.empty(%dim) : tensor<?x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%input, %scale : tensor<?x8xi8>, tensor<?xf32>)
      outs(%init : tensor<?x8xf32>) -> tensor<?x8xf32>
  util.return %dequantized : tensor<?x8xf32>
}
// CHECK-LABEL: util.func public @decline_dynamic_shape(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<?x8xi8>, %[[SCALE:[a-zA-Z0-9_]+]]: tensor<?xf32>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[INPUT]], %[[SCALE]] : tensor<?x8xi8>, tensor<?xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// An empty tensor has nothing to pack.
util.func public @decline_zero_size(%scale: f32) -> tensor<4x0xf32> {
  %weight = arith.constant dense<0> : tensor<4x0xi8>
  %init = tensor.empty() : tensor<4x0xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x0xi8>, f32)
      outs(%init : tensor<4x0xf32>) -> tensor<4x0xf32>
  util.return %dequantized : tensor<4x0xf32>
}
// CHECK-LABEL: util.func public @decline_zero_size(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: f32
//       CHECK:   %[[WEIGHT:.+]] = arith.constant dense<0> : tensor<4x0xi8>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x0xi8>, f32)
//       CHECK:   util.return %[[DEQ]]

//===----------------------------------------------------------------------===//
// Readers of the input
//
// Each input is packed once, right after its definition, at the widest width
// its dequantizes need. An input that another op still reads as i8, also
// through another load of the same global, is not packed, since it would be
// stored twice.
//===----------------------------------------------------------------------===//

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The packing goes next to the load, outside the scf.if, so constant
// evaluation can fold it.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @pack_outside_control_flow(%cond: i1, %scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %result = scf.if %cond -> tensor<4x8xf32> {
    %dequantized = iree_linalg_ext.dequantize_affine
        <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
        ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
        outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
    scf.yield %dequantized : tensor<4x8xf32>
  } else {
    scf.yield %init : tensor<4x8xf32>
  }
  util.return %result : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_outside_control_flow(
//  CHECK-SAME:     %[[COND:[a-zA-Z0-9_]+]]: i1, %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[RESULT:.+]] = scf.if %[[COND]]
//       CHECK:     %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:       ins(%[[NARROW]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:     scf.yield %[[DEQ]]
//       CHECK:   util.return %[[RESULT]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// One dequantize reads the weight with a 2-bit range and one with a 4-bit
// range. The weight is packed once, into i4, which holds both ranges.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @pack_once_at_widest_width(%scale: tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %two_bit = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -2, quant_max = 1>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  %four_bit = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %two_bit, %four_bit : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_once_at_widest_width(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[TWO_BIT:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -2, quant_max = 1>
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:   %[[FOUR_BIT:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:   util.return %[[TWO_BIT]], %[[FOUR_BIT]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The second dequantize reads the weight with an 8-bit range and keeps reading
// it as i8. Packing it for the first one would store the weight twice.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_input_read_unpacked(%scale: tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized0 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  %dequantized1 = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -128, quant_max = 127>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized0, %dequantized1 : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_input_read_unpacked(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[DEQ0:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   %[[DEQ1:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ0]], %[[DEQ1]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Both functions dequantize the weight, so both loads are packed.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @pack_global_dequantized_twice(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
util.func public @dequantize_weight_again(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_global_dequantized_twice(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]
// CHECK-LABEL: util.func public @dequantize_weight_again(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[NARROW:.+]] = flow.tensor.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Another function loads the weight and returns it as i8. Packing it for the
// dequantize would store the weight twice.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_global_read_unpacked(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
util.func public @read_weight() -> tensor<4x8xi8> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  util.return %weight : tensor<4x8xi8>
}
// CHECK-LABEL: util.func public @decline_global_read_unpacked(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

//===----------------------------------------------------------------------===//
// Where the input comes from
//
// Only constant expressions of inline constants or immutable globals with
// inline data are packed, so that constant evaluation folds the packing. A
// runtime value would be packed on every call, and a parameter at every load
// while its archive keeps the unpacked bytes.
//===----------------------------------------------------------------------===//

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// A function argument is a runtime value.
util.func public @decline_runtime_value(%input: tensor<4x8xi8>, %scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%input, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_runtime_value(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[INPUT]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The weight's initial value is a parameter rather than inline data.
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
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

//===----------------------------------------------------------------------===//
// Storage type
//
// Quantized values arrive in i8. Wider storage would need a truncation before
// packing, and sub-byte storage is already packed.
//===----------------------------------------------------------------------===//

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Packed i4 storage, as the pass itself produces, is left unchanged.
util.global private @weight = dense<1> : tensor<4x8xi4>
util.func public @decline_i4_storage(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi4>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi4>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_i4_storage(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi4>
//   CHECK-NOT:   flow.tensor.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]
