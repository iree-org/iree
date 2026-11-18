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
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %[[PACKED]] : tensor<4x2xi8> -> tensor<4x8xi2>
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
//   CHECK-NOT:   iree_tensor_ext.bitcast
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
//   CHECK-NOT:   iree_tensor_ext.bitcast
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
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
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
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
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
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
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
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
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
//   CHECK-NOT:   iree_tensor_ext.bitcast
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
//   CHECK-NOT:   iree_tensor_ext.bitcast
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
util.func public @pack_i2_range_into_i4_for_row_size(%scale: tensor<4xf32>) -> tensor<4x6xf32> {
  %weight = util.global.load @weight : tensor<4x6xi8>
  %init = tensor.empty() : tensor<4x6xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -2, quant_max = 1>
      ins(%weight, %scale : tensor<4x6xi8>, tensor<4xf32>)
      outs(%init : tensor<4x6xf32>) -> tensor<4x6xf32>
  util.return %dequantized : tensor<4x6xf32>
}
// CHECK-LABEL: util.func public @pack_i2_range_into_i4_for_row_size(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x3xi8> -> tensor<4x6xi4>
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
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[INPUT]], %[[SCALE]] : tensor<?x8xi8>, tensor<?xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// An empty tensor has nothing to pack.
util.func public @decline_empty_tensor(%scale: f32) -> tensor<4x0xf32> {
  %weight = arith.constant dense<0> : tensor<4x0xi8>
  %init = tensor.empty() : tensor<4x0xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x0xi8>, f32)
      outs(%init : tensor<4x0xf32>) -> tensor<4x0xf32>
  util.return %dequantized : tensor<4x0xf32>
}
// CHECK-LABEL: util.func public @decline_empty_tensor(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: f32
//       CHECK:   %[[WEIGHT:.+]] = arith.constant dense<0> : tensor<4x0xi8>
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x0xi8>, f32)
//       CHECK:   util.return %[[DEQ]]

//===----------------------------------------------------------------------===//
// Readers of the input
//
// Each input is packed once, right after its definition, at the widest width
// the dequantizes of its data need, the same for every load of a global. An
// input that another op still reads as i8, also through a reshape or another
// load of the same global, that dequantizes read in different shapes, or that
// is a public global, is not packed, since it would be stored twice.
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
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[RESULT:.+]] = scf.if %[[COND]]
//       CHECK:     %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:       ins(%[[NARROW]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:     scf.yield %[[DEQ]]
//       CHECK:   util.return %[[RESULT]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The load is inside the scf.if, so the packing next to it would not be
// hoisted and would run on every call.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_input_in_control_flow(%cond: i1, %scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %init = tensor.empty() : tensor<4x8xf32>
  %result = scf.if %cond -> tensor<4x8xf32> {
    %weight = util.global.load @weight : tensor<4x8xi8>
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
// CHECK-LABEL: util.func public @decline_input_in_control_flow(
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   scf.if
//       CHECK:     %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//   CHECK-NOT:     iree_tensor_ext.bitcast
//       CHECK:     iree_linalg_ext.dequantize_affine
//  CHECK-SAME:       ins(%[[WEIGHT]], %{{.+}} : tensor<4x8xi8>, tensor<4xf32>)

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
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//   CHECK-NOT:   iree_tensor_ext.bitcast
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

// Each function loads the weight, one for a 2-bit and one for a 4-bit range.
// Both loads pack into i4, so the packings are identical and merge later.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @pack_loads_at_widest_width(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -2, quant_max = 1>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
util.func public @dequantize_four_bit(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_loads_at_widest_width(
//       CHECK:   iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -2, quant_max = 1>
// CHECK-LABEL: util.func public @dequantize_four_bit(
//       CHECK:   iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The second dequantize reads the weight as its zero point, which stays i8.
// Packing the weight for the first one would store it twice. The second
// dequantize's own input is packed.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.global private @other_weight = dense<2> : tensor<4x8xi8>
util.func public @decline_input_read_as_zero_point(%scale: tensor<4xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %other_weight = util.global.load @other_weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  %other = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2, #id2], quant_min = -8, quant_max = 7>
      ins(%other_weight, %scale, %weight : tensor<4x8xi8>, tensor<4xf32>, tensor<4x8xi8>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized, %other : tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_input_read_as_zero_point(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//       CHECK:   %[[OTHER_WEIGHT:.+]] = util.global.load @other_weight : tensor<4x8xi8>
//       CHECK:   tensor.extract_slice %[[OTHER_WEIGHT]]
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   %[[OTHER:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]], %[[WEIGHT]] : tensor<4x8xi4>, tensor<4xf32>, tensor<4x8xi8>)
//       CHECK:   util.return %[[DEQ]], %[[OTHER]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Both functions dequantize the weight in the same shape, so both loads are
// packed.
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
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]
// CHECK-LABEL: util.func public @dequantize_weight_again(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#row3 = affine_map<(d0, d1, d2) -> (d0)>

// The second function dequantizes the weight through a reshape. Packing it in
// both shapes would store it twice.
util.global private @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_global_read_in_two_shapes(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
util.func public @dequantize_reshaped_weight(%scale: tensor<4xf32>) -> tensor<4x2x4xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %expanded = tensor.expand_shape %weight [[0], [1, 2]] output_shape [4, 2, 4]
      : tensor<4x8xi8> into tensor<4x2x4xi8>
  %init = tensor.empty() : tensor<4x2x4xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id3, #row3, #id3], quant_min = -8, quant_max = 7>
      ins(%expanded, %scale : tensor<4x2x4xi8>, tensor<4xf32>)
      outs(%init : tensor<4x2x4xf32>) -> tensor<4x2x4xf32>
  util.return %dequantized : tensor<4x2x4xf32>
}
// CHECK-LABEL: util.func public @decline_global_read_in_two_shapes(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]
// CHECK-LABEL: util.func public @dequantize_reshaped_weight(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[EXPANDED]], %[[SCALE]] : tensor<4x2x4xi8>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Another function loads the weight and returns it as i8, while the dequantize
// reads it through a reshape. Packing it for the dequantize would store the
// weight twice.
util.global private @weight = dense<1> : tensor<4x2x4xi8>
util.func public @decline_global_read_unpacked(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x2x4xi8>
  %collapsed = tensor.collapse_shape %weight [[0], [1, 2]]
      : tensor<4x2x4xi8> into tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%collapsed, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
util.func public @read_weight() -> tensor<4x2x4xi8> {
  %weight = util.global.load @weight : tensor<4x2x4xi8>
  util.return %weight : tensor<4x2x4xi8>
}
// CHECK-LABEL: util.func public @decline_global_read_unpacked(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x2x4xi8>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[WEIGHT]]
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[COLLAPSED]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Code outside the module can read a public global, so it stays.
util.global public @weight = dense<1> : tensor<4x8xi8>
util.func public @decline_public_global(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @decline_public_global(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<4x8xi8>
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

//===----------------------------------------------------------------------===//
// Where the input comes from
//
// Only constant expressions of inline constants or immutable globals with
// inline data are packed, so that constant evaluation folds the packing. A
// runtime value would be packed on every call, and a parameter at every load
// while its archive keeps the unpacked bytes. Hoisting rejects an expression
// that grows by more than a threshold the pass cannot see, so each packing
// stores no more bytes than leaving its input unpacked.
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
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     quant_min = -8, quant_max = 7>
//  CHECK-SAME:     ins(%[[INPUT]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#byte3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// MatMulNBits stores two 4-bit values per byte, and its lowering unpacks them
// into i8. Packing them again takes exactly the bytes of the stored data.
util.global private @packed_weight = dense<33> : tensor<4x4xi8>
util.func public @pack_unpacked_nibbles(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %packed_weight = util.global.load @packed_weight : tensor<4x4xi8>
  %empty = tensor.empty() : tensor<4x4x2xi8>
  %nibbles = linalg.generic
      {indexing_maps = [#byte3, #id3],
       iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%packed_weight : tensor<4x4xi8>) outs(%empty : tensor<4x4x2xi8>) {
  ^bb0(%byte: i8, %out: i8):
    %c4 = arith.constant 4 : i8
    %c15 = arith.constant 15 : i8
    %lane = linalg.index 2 : index
    %lane_i8 = arith.index_cast %lane : index to i8
    %shift = arith.muli %lane_i8, %c4 : i8
    %shifted = arith.shrui %byte, %shift : i8
    %nibble = arith.andi %shifted, %c15 : i8
    linalg.yield %nibble : i8
  } -> tensor<4x4x2xi8>
  %weight = tensor.collapse_shape %nibbles [[0], [1, 2]]
      : tensor<4x4x2xi8> into tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = 0, quant_max = 15,
       input_unsigned>
      ins(%weight, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @pack_unpacked_nibbles(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[WEIGHT:.+]] = tensor.collapse_shape
//       CHECK:   tensor.extract_slice %[[WEIGHT]]
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// The broadcast of two bytes packs into eight, more than the data it is
// computed from.
util.func public @decline_packing_larger_than_data(%scale: f32) -> tensor<2x8xf32> {
  %values = arith.constant dense<[1, -1]> : tensor<2xi8>
  %empty = tensor.empty() : tensor<2x8xi8>
  %broadcast = linalg.broadcast ins(%values : tensor<2xi8>)
      outs(%empty : tensor<2x8xi8>) dimensions = [1]
  %init = tensor.empty() : tensor<2x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #id2], quant_min = -8, quant_max = 7>
      ins(%broadcast, %scale : tensor<2x8xi8>, f32)
      outs(%init : tensor<2x8xf32>) -> tensor<2x8xf32>
  util.return %dequantized : tensor<2x8xf32>
}
// CHECK-LABEL: util.func public @decline_packing_larger_than_data(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: f32
//       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[BROADCAST]], %[[SCALE]] : tensor<2x8xi8>, f32)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Another function reads the weight, so it stays. Unpacked, hoisting stores the
// transpose in a global of its own, which packing shrinks.
util.global private @weight = dense<1> : tensor<8x4xi8>
util.func public @pack_transpose_of_shared_weight(%scale: tensor<4xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<8x4xi8>
  %empty = tensor.empty() : tensor<4x8xi8>
  %transposed = linalg.transpose ins(%weight : tensor<8x4xi8>)
      outs(%empty : tensor<4x8xi8>) permutation = [1, 0]
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%transposed, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %dequantized : tensor<4x8xf32>
}
util.func public @read_weight() -> tensor<8x4xi8> {
  %weight = util.global.load @weight : tensor<8x4xi8>
  util.return %weight : tensor<8x4xi8>
}
// CHECK-LABEL: util.func public @pack_transpose_of_shared_weight(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = linalg.transpose
//       CHECK:   tensor.extract_slice %[[TRANSPOSED]]
//       CHECK:   %[[NARROW:.+]] = iree_tensor_ext.bitcast %{{.+}} : tensor<4x4xi8> -> tensor<4x8xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[NARROW]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The second dequantize reads the weight that the slice also reads unpacked,
// so it is not packed and keeps the weight. Unpacked, the slice is read in
// place, so packing it would add its bytes to the weight.
util.global private @weight = dense<1> : tensor<8x8xi8>
util.func public @decline_slice_of_kept_weight(%scale: tensor<4xf32>, %weight_scale: tensor<8xf32>) -> (tensor<4x8xf32>, tensor<8x8xf32>) {
  %weight = util.global.load @weight : tensor<8x8xi8>
  %slice = tensor.extract_slice %weight[0, 0] [4, 8] [1, 1]
      : tensor<8x8xi8> to tensor<4x8xi8>
  %init = tensor.empty() : tensor<4x8xf32>
  %dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%slice, %scale : tensor<4x8xi8>, tensor<4xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  %weight_init = tensor.empty() : tensor<8x8xf32>
  %weight_dequantized = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %weight_scale : tensor<8x8xi8>, tensor<8xf32>)
      outs(%weight_init : tensor<8x8xf32>) -> tensor<8x8xf32>
  util.return %dequantized, %weight_dequantized : tensor<4x8xf32>, tensor<8x8xf32>
}
// CHECK-LABEL: util.func public @decline_slice_of_kept_weight(
//  CHECK-SAME:     %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4xf32>, %[[WEIGHT_SCALE:[a-zA-Z0-9_]+]]: tensor<8xf32>
//       CHECK:   %[[WEIGHT:.+]] = util.global.load @weight : tensor<8x8xi8>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[WEIGHT]]
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[SLICE]], %[[SCALE]] : tensor<4x8xi8>, tensor<4xf32>)
//       CHECK:   %[[WEIGHT_DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[WEIGHT_SCALE]] : tensor<8x8xi8>, tensor<8xf32>)
//       CHECK:   util.return %[[DEQ]], %[[WEIGHT_DEQ]]

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
//   CHECK-NOT:   iree_tensor_ext.bitcast
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
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<4x8xi4>, tensor<4xf32>)
//       CHECK:   util.return %[[DEQ]]
