// RUN: iree-opt --iree-dispatch-creation-elementwise-op-fusion --split-input-file --mlir-print-local-scope  %s | FileCheck %s

util.func public @transpose_attention(%arg0: tensor<4x64x32x128xf16>, %arg1: tensor<4x64x32x128xf16>, %arg2: tensor<4x64x32x128xf16>, %arg3: f16) -> tensor<4x64x4096xf16> {
  %0 = tensor.empty() : tensor<4x32x64x128xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %4 = tensor.empty() : tensor<4x32x64x128xf16>
  %5 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]} ins(%1, %2, %3, %arg3 : tensor<4x32x64x128xf16>, tensor<4x32x64x128xf16>, tensor<4x32x64x128xf16>, f16) outs(%4 : tensor<4x32x64x128xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<4x32x64x128xf16>
  %6 = tensor.empty() : tensor<4x64x32x128xf16>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<4x32x64x128xf16>) outs(%6 : tensor<4x64x32x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x64x32x128xf16>
  %collapsed = tensor.collapse_shape %7 [[0], [1], [2, 3]] : tensor<4x64x32x128xf16> into tensor<4x64x4096xf16>
  util.return %collapsed : tensor<4x64x4096xf16>
}
// CHECK-LABEL: util.func public @transpose_attention
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG3:[A-Za-z0-9]+]]: f16
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d5)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
//  CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]

// -----

util.func public @transposed_attention_masked(%arg0: tensor<4x64x32x128xf16>, %arg1: tensor<4x64x32x128xf16>, %arg2: tensor<4x64x32x128xf16>, %arg3: f16, %arg4: tensor<4x64x32x64xf16>) -> tensor<4x64x4096xf16> {
  %0 = tensor.empty() : tensor<4x32x64x128xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<4x64x32x128xf16>) outs(%0 : tensor<4x32x64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x128xf16>
  %empty = tensor.empty() : tensor<4x32x64x64xf16>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg4 : tensor<4x64x32x64xf16>) outs(%empty : tensor<4x32x64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x64x64xf16>
  %5 = tensor.empty() : tensor<4x32x64x128xf16>
  %6 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]} ins(%1, %2, %3, %arg3, %4 : tensor<4x32x64x128xf16>, tensor<4x32x64x128xf16>, tensor<4x32x64x128xf16>, f16, tensor<4x32x64x64xf16>) outs(%5 : tensor<4x32x64x128xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<4x32x64x128xf16>
  %7 = tensor.empty() : tensor<4x64x32x128xf16>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<4x32x64x128xf16>) outs(%7 : tensor<4x64x32x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x64x32x128xf16>
  %collapsed = tensor.collapse_shape %8 [[0], [1], [2, 3]] : tensor<4x64x32x128xf16> into tensor<4x64x4096xf16>
  util.return %collapsed : tensor<4x64x4096xf16>
}
// CHECK-LABEL: util.func public @transposed_attention_masked
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG3:[A-Za-z0-9]+]]: f16
//  CHECK-SAME:   %[[ARG4:[A-Za-z0-9]+]]: tensor
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d1, d5)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d4)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
//  CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]

// -----

util.func public @transpose_matmul(%arg0 : tensor<100x100xf16>, %arg1 : tensor<100x100xf16>) -> (tensor<100x100xf16>) {
  %0 = tensor.empty() : tensor<100x100xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<100x100xf16>) outs(%0 : tensor<100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<100x100xf16>
  %2 = tensor.empty() : tensor<100x100xf16>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<100x100xf16>) outs(%2 : tensor<100x100xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<100x100xf16>
  %5 = tensor.empty() : tensor<100x100xf16>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1, %3: tensor<100x100xf16>, tensor<100x100xf16>) outs(%5 : tensor<100x100xf16>) {
  ^bb0(%in: f16, %in_0 : f16, %out: f16):
    %01 = arith.mulf %in, %in_0 : f16
    %02 = arith.addf %01, %out: f16
    linalg.yield %02  : f16
  } -> tensor<100x100xf16>
  util.return  %4 : tensor<100x100xf16>
}
// CHECK-LABEL: util.func public @transpose_matmul
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//       CHECK:   %[[RET:.+]] = linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2) -> (d2, d0)>
//  CHECK-SAME:     affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d1)>
//  CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]]

// -----

util.func public @fuse_generic_gather(
  %11 :tensor<128256x4096xf16>, %12 : tensor<4x?xi64>,
  %13 : tensor<4x?x4096xf32>, %14 : tensor<128256x4096xf32>)
    -> tensor<4x?x4096xf32>{

  %15 = linalg.generic {
    indexing_maps = [ affine_map<(d0, d1) -> (d0, d1)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%11 : tensor<128256x4096xf16>)
    outs(%14 : tensor<128256x4096xf32>) {
      ^bb0(%in: f16, %out: f32):
        %17 = arith.extf %in : f16 to f32
        linalg.yield %17 : f32
    } -> tensor<128256x4096xf32>
  %16 = linalg.generic {
    indexing_maps = [ affine_map<(d0, d1, d2) -> (d0, d1)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%12 : tensor<4x?xi64>)
    outs(%13 : tensor<4x?x4096xf32>) {
      ^bb0(%in: i64, %out: f32):
        %17 = arith.index_cast %in : i64 to index
        %18 = linalg.index 2 : index
        %extracted = tensor.extract %15[%17, %18] : tensor<128256x4096xf32>
        linalg.yield %extracted : f32
      } -> tensor<4x?x4096xf32>
  util.return %16 : tensor<4x?x4096xf32>
}
// CHECK:         %[[INDEX0:[a-zA-Z0-9]+]] = arith.index_cast %in : i64 to index
// CHECK:         %[[INDEX1:[a-zA-Z0-9]+]] = linalg.index 2 : index
// CHECK-NEXT:    %[[EXTRACTED:.*]] = tensor.extract %[[TENSOR0:.+]][%[[INDEX0]], %[[INDEX1]]] : tensor<128256x4096xf16>
// CHECK-NEXT:    %[[RES:[a-zA-Z0-9]+]] = arith.extf %[[EXTRACTED]] : f16 to f32
// CHECK-NEXT:    linalg.yield %[[RES]] : f32


// -----

util.func public @fuse_generic_gather2(
  %11 :tensor<128256x4096xf16>, %12 : tensor<4x?xi64>,
  %13 : tensor<4x?x4096xf32>, %14 : tensor<128256x4096xf32>)
    -> tensor<4x?x4096xf32>{

  %15 = linalg.generic {
    indexing_maps = [ affine_map<(d0, d1) -> (d0, d1)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%11 : tensor<128256x4096xf16>)
    outs(%14 : tensor<128256x4096xf32>) {
      ^bb0(%in: f16, %out: f32):
        %17 = arith.extf %in : f16 to f32
        linalg.yield %17 : f32
    } -> tensor<128256x4096xf32>
  %16 = linalg.generic {
    indexing_maps = [ affine_map<(d0, d1, d2) -> (d0, d1)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%12 : tensor<4x?xi64>)
    outs(%13 : tensor<4x?x4096xf32>) {
      ^bb0(%in: i64, %out: f32):
        %17 = arith.index_cast %in : i64 to index
        %18 = linalg.index 2 : index
        %extracted = tensor.extract %15[%17, %18] : tensor<128256x4096xf32>
        %result = arith.addf %extracted, %extracted : f32
        %result2 = arith.mulf %extracted, %extracted : f32
        %final = arith.addf %result, %result2 : f32
        linalg.yield %final: f32
      } -> tensor<4x?x4096xf32>
  util.return %16 : tensor<4x?x4096xf32>
}
// CHECK:         %[[INDEX0:[a-zA-Z0-9]+]] = arith.index_cast %in : i64 to index
// CHECK:         %[[INDEX1:[a-zA-Z0-9]+]] = linalg.index 2 : index
// CHECK-NEXT:    %[[EXTRACTED:.*]] = tensor.extract %[[TENSOR0:.+]][%[[INDEX0]], %[[INDEX1]]] : tensor<128256x4096xf16>
// CHECK-NEXT:    %[[RES:[a-zA-Z0-9]+]] = arith.extf %[[EXTRACTED]] : f16 to f32
// CHECK-NEXT:    %[[RES2:[a-zA-Z0-9]+]] = arith.addf %[[RES]], %[[RES]] : f32
// CHECK-NEXT:    %[[RES3:[a-zA-Z0-9]+]] = arith.mulf %[[RES]], %[[RES]] : f32
// CHECK-NEXT:    %[[RES4:[a-zA-Z0-9]+]] = arith.addf %[[RES2]], %[[RES3]] : f32
// CHECK-NEXT:    linalg.yield %[[RES4]] : f32

// -----

util.func public @fuse_transpose_attention_to_producer(%q: tensor<2x10x4096x64xf16>, %k: tensor<2x10x4096x64xf16>, %quantized_v: tensor<2x10x4096x64xi32>, %quant_offset: tensor<10x64xi32>, %quant_scale: tensor<10x64xf32>, %scale: f16) -> tensor<2x10x4096x64xf16> {
  // Dequantize int-quantization of V
  %init_dequant = tensor.empty() : tensor<2x10x4096x64xf16>
  %v = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%quantized_v, %quant_offset, %quant_scale : tensor<2x10x4096x64xi32>, tensor<10x64xi32>, tensor<10x64xf32>) outs(%init_dequant : tensor<2x10x4096x64xf16>) {
  ^bb0(%in: i32, %in_0: i32, %in_1: f32, %out: f16):
      %19 = arith.addi %in, %in_0 : i32
      %20 = arith.sitofp %19 : i32 to f32
      %21 = arith.mulf %20, %in_1 : f32
      %22 = arith.truncf %21 : f32 to f16
      linalg.yield %22 : f16
  } -> tensor<2x10x4096x64xf16>

  // Transpose-V
  %init_transpose = tensor.empty() : tensor<2x10x64x4096xf16>
  %transpose_v = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%v : tensor<2x10x4096x64xf16>) outs(%init_transpose : tensor<2x10x64x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<2x10x64x4096xf16>

  // Attention-Transpose-V
  %init_attention = tensor.empty() : tensor<2x10x4096x64xf16>
  %attention = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]} ins(%q, %k, %transpose_v, %scale : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x64x4096xf16>, f16) outs(%init_attention : tensor<2x10x4096x64xf16>) {
    ^bb0(%score: f16):
      iree_linalg_ext.yield %score: f16
  } -> tensor<2x10x4096x64xf16>
  util.return %attention : tensor<2x10x4096x64xf16>
}
// CHECK-LABEL: util.func public @fuse_transpose_attention_to_producer
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG3:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG4:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG5:[A-Za-z0-9]+]]: f16
//       CHECK:   %[[DEQUANT_V:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-SAME:      affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-SAME:      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>]
//  CHECK-SAME:     ins(%[[ARG2]], %[[ARG3]], %[[ARG4]]
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
//  CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]], %[[DEQUANT_V]], %[[ARG5]]

// -----

util.func public @fuse_attention_with_broadcast(%arg0: tensor<4x8x128x?xf16>, %arg1: tensor<4x8x4x?x32x128xf16>, %arg2: tensor<4x8x4x?x128xf16>, %arg3: f16, %arg4: tensor<4x8x4x?x32x?xf16>, %arg5: tensor<4x8x4x?x32x128xf16>, %arg6: tensor<4x8x4x128x?xf16>) -> tensor<4x8x4x?x32x128xf16> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x8x128x?xf16>) outs(%arg6 : tensor<4x8x4x128x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x8x4x128x?xf16>
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d7, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d5, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>]} ins(%arg1, %arg2, %0, %arg3, %arg4 : tensor<4x8x4x?x32x128xf16>, tensor<4x8x4x?x128xf16>, tensor<4x8x4x128x?xf16>, f16, tensor<4x8x4x?x32x?xf16>) outs(%arg5 : tensor<4x8x4x?x32x128xf16>) {
  ^bb0(%arg7: f32):
    iree_linalg_ext.yield %arg7 : f32
  } -> tensor<4x8x4x?x32x128xf16>
  util.return %1 : tensor<4x8x4x?x32x128xf16>
}
// CHECK-LABEL: func public @fuse_attention_with_broadcast
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]:
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d6)>,
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d7, d6)>,
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d7)>,
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>,
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d7)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>
//  CHECK-SAME:       ins(%[[ARG1]], %[[ARG2]], %[[ARG0]], %[[ARG3]], %[[ARG4]] :
//       CHECK:   util.return %[[ATTENTION]]


// -----

util.func public @fuse_attention_with_broadcast_transpose(%arg0: tensor<4x?x8x128xf16>, %arg1: tensor<4x8x4x?x32x128xf16>, %arg2: tensor<4x8x4x?x128xf16>, %arg3: f16, %arg4: tensor<4x8x4x?x32x?xf16>, %arg5: tensor<4x8x4x?x32x128xf16>, %arg6: tensor<4x8x4x128x?xf16>) -> tensor<4x8x4x?x32x128xf16> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<4x?x8x128xf16>) outs(%arg6 : tensor<4x8x4x128x?xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x8x4x128x?xf16>
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d7, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d5, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>]} ins(%arg1, %arg2, %0, %arg3, %arg4 : tensor<4x8x4x?x32x128xf16>, tensor<4x8x4x?x128xf16>, tensor<4x8x4x128x?xf16>, f16, tensor<4x8x4x?x32x?xf16>) outs(%arg5 : tensor<4x8x4x?x32x128xf16>) {
  ^bb0(%arg7: f32):
    iree_linalg_ext.yield %arg7 : f32
  } -> tensor<4x8x4x?x32x128xf16>
  util.return %1 : tensor<4x8x4x?x32x128xf16>
}
// CHECK-LABEL: func public @fuse_attention_with_broadcast_transpose
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]:
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]:
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d6)>,
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d7, d6)>,
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d7, d1, d5)>,
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>,
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d7)>
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5)>
//  CHECK-SAME:       ins(%[[ARG1]], %[[ARG2]], %[[ARG0]], %[[ARG3]], %[[ARG4]] :
//       CHECK:   util.return %[[ATTENTION]]

// -----

util.func public @gather_fusion(%arg0: tensor<2x64x64x640xf16>, %arg1: tensor<2x64x64x640xf16>, %arg2: tensor<2xi64>, %arg3: tensor<640xi64>, %arg4: tensor<128xi64>, %arg5: tensor<640xf16>, %arg6: tensor<f32>) -> tensor<2x128x128x640xi8> {
  %cst = arith.constant -1.280000e+02 : f16
  %cst_0 = arith.constant 1.270000e+02 : f16
  %0 = tensor.empty() : tensor<2x128x128x640xi8>
  %1 = tensor.empty() : tensor<2x640x64x64xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x64x64x640xf16>, tensor<2x64x64x640xf16>) outs(%1 : tensor<2x640x64x64xf32>) {
  ^bb0(%in: f16, %in_1: f16, %out: f32):
    %4 = arith.addf %in, %in_1 : f16
    %5 = arith.extf %4 : f16 to f32
    linalg.yield %5 : f32
  } -> tensor<2x640x64x64xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d2)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg3, %arg4, %arg4, %arg5, %arg6 : tensor<2xi64>, tensor<640xi64>, tensor<128xi64>, tensor<128xi64>, tensor<640xf16>, tensor<f32>) outs(%0 : tensor<2x128x128x640xi8>) {
  ^bb0(%in: i64, %in_1: i64, %in_2: i64, %in_3: i64, %in_4: f16, %in_5: f32, %out: i8):
    %4 = arith.index_cast %in : i64 to index
    %5 = arith.index_cast %in_1 : i64 to index
    %6 = arith.index_cast %in_2 : i64 to index
    %7 = arith.index_cast %in_3 : i64 to index
    %extracted = tensor.extract %2[%4, %5, %6, %7] : tensor<2x640x64x64xf32>
    %8 = arith.truncf %extracted : f32 to f16
    %9 = arith.mulf %8, %in_4 : f16
    %10 = arith.truncf %in_5 : f32 to f16
    %11 = arith.divf %9, %10 : f16
    %12 = math.roundeven %11 : f16
    %13 = arith.cmpf ult, %12, %cst : f16
    %14 = arith.select %13, %cst, %12 : f16
    %15 = arith.cmpf ugt, %14, %cst_0 : f16
    %16 = arith.select %15, %cst_0, %14 : f16
    %17 = arith.fptosi %16 : f16 to i8
    linalg.yield %17 : i8
  } -> tensor<2x128x128x640xi8>
  util.return %3 : tensor<2x128x128x640xi8>
}
// CHECK-LABEL: util.func public @gather_fusion(
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG3:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG4:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG5:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG6:[A-Za-z0-9]+]]: tensor
//       CHECK:   %[[GEN:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d3)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d1)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d2)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d3)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> ()>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//  CHECK-SAME:     ins(%[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG4]], %[[ARG5]], %[[ARG6]]
//       CHECK:     ^bb0(
//  CHECK-SAME:       %[[IN0:[_a-zA-Z0-9]+]]: i64,
//  CHECK-SAME:       %[[IN1:[_a-zA-Z0-9]+]]: i64,
//  CHECK-SAME:       %[[IN2:[_a-zA-Z0-9]+]]: i64,
//  CHECK-SAME:       %[[IN3:[_a-zA-Z0-9]+]]: i64,
//   CHECK-DAG:     %[[CAST0:.+]] = arith.index_cast %[[IN0]] : i64 to index
//   CHECK-DAG:     %[[CAST1:.+]] = arith.index_cast %[[IN1]] : i64 to index
//   CHECK-DAG:     %[[CAST2:.+]] = arith.index_cast %[[IN2]] : i64 to index
//   CHECK-DAG:     %[[CAST3:.+]] = arith.index_cast %[[IN3]] : i64 to index
//       CHECK:     %[[EXTRACT0:.*]] = tensor.extract %[[ARG0]][%[[CAST0]], %[[CAST2]], %[[CAST3]], %[[CAST1]]] : tensor<2x64x64x640xf16>
//       CHECK:     %[[EXTRACT1:.*]] = tensor.extract %[[ARG1]][%[[CAST0]], %[[CAST2]], %[[CAST3]], %[[CAST1]]] : tensor<2x64x64x640xf16>
//       CHECK:     %[[ADDF:.+]] = arith.addf %[[EXTRACT0]], %[[EXTRACT1]] : f16
//       CHECK:   util.return %[[GEN]] : tensor<2x128x128x640xi8>

// -----

util.func public @gather_fusion_compose_maps(%arg0: tensor<2x64x64x640xf16>, %arg1: tensor<2x64x64x640xf16>, %arg2: tensor<2xi64>, %arg3: tensor<640xi64>, %arg4: tensor<128xi64>, %arg5: tensor<640xf16>, %arg6: tensor<f32>) -> tensor<2x128x128x640xi8> {
  %cst = arith.constant -1.280000e+02 : f16
  %cst_0 = arith.constant 1.270000e+02 : f16
  %0 = tensor.empty() : tensor<2x128x128x640xi8>
  %1 = tensor.empty() : tensor<2x640x64x64xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x64x64x640xf16>, tensor<2x64x64x640xf16>) outs(%1 : tensor<2x640x64x64xf32>) {
  ^bb0(%in: f16, %in_1: f16, %out: f32):
    %4 = arith.addf %in, %in_1 : f16
    %5 = arith.extf %4 : f16 to f32
    linalg.yield %5 : f32
  } -> tensor<2x640x64x64xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d2)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg3, %arg4, %arg4, %arg5, %arg6 : tensor<2xi64>, tensor<640xi64>, tensor<128xi64>, tensor<128xi64>, tensor<640xf16>, tensor<f32>) outs(%0 : tensor<2x128x128x640xi8>) {
  ^bb0(%in: i64, %in_1: i64, %in_2: i64, %in_3: i64, %in_4: f16, %in_5: f32, %out: i8):
    %4 = arith.index_cast %in : i64 to index
    %5 = arith.index_cast %in_1 : i64 to index
    %6 = arith.index_cast %in_2 : i64 to index
    %7 = arith.index_cast %in_3 : i64 to index
    %extracted = tensor.extract %2[%4, %5, %6, %7] : tensor<2x640x64x64xf32>
    %8 = arith.truncf %extracted : f32 to f16
    %9 = arith.mulf %8, %in_4 : f16
    %10 = arith.truncf %in_5 : f32 to f16
    %11 = arith.divf %9, %10 : f16
    %12 = math.roundeven %11 : f16
    %13 = arith.cmpf ult, %12, %cst : f16
    %14 = arith.select %13, %cst, %12 : f16
    %15 = arith.cmpf ugt, %14, %cst_0 : f16
    %16 = arith.select %15, %cst_0, %14 : f16
    %17 = arith.fptosi %16 : f16 to i8
    linalg.yield %17 : i8
  } -> tensor<2x128x128x640xi8>
  util.return %3 : tensor<2x128x128x640xi8>
}
// CHECK-LABEL: util.func public @gather_fusion_compose_maps(
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG3:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG4:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG5:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG6:[A-Za-z0-9]+]]: tensor
//       CHECK:   %[[GEN:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d3)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d1)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d2)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d3)>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> ()>,
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//  CHECK-SAME:     ins(%[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG4]], %[[ARG5]], %[[ARG6]]
//       CHECK:     ^bb0(
//  CHECK-SAME:       %[[IN0:[_a-zA-Z0-9]+]]: i64,
//  CHECK-SAME:       %[[IN1:[_a-zA-Z0-9]+]]: i64,
//  CHECK-SAME:       %[[IN2:[_a-zA-Z0-9]+]]: i64,
//  CHECK-SAME:       %[[IN3:[_a-zA-Z0-9]+]]: i64,
//   CHECK-DAG:     %[[CAST0:.+]] = arith.index_cast %[[IN0]] : i64 to index
//   CHECK-DAG:     %[[CAST1:.+]] = arith.index_cast %[[IN1]] : i64 to index
//   CHECK-DAG:     %[[CAST2:.+]] = arith.index_cast %[[IN2]] : i64 to index
//   CHECK-DAG:     %[[CAST3:.+]] = arith.index_cast %[[IN3]] : i64 to index
//       CHECK:     %[[EXTRACT0:.*]] = tensor.extract %[[ARG0]][%[[CAST0]], %[[CAST2]], %[[CAST3]], %[[CAST1]]] : tensor<2x64x64x640xf16>
//       CHECK:     %[[EXTRACT1:.*]] = tensor.extract %[[ARG1]][%[[CAST0]], %[[CAST3]], %[[CAST2]], %[[CAST1]]] : tensor<2x64x64x640xf16>
//       CHECK:     %[[ADDF:.+]] = arith.addf %[[EXTRACT0]], %[[EXTRACT1]] : f16
//       CHECK:   util.return %[[GEN]] : tensor<2x128x128x640xi8>

// -----

util.func public @gather_0d_producer(%arg0 : tensor<f16>, %arg1 : tensor<100xindex>, %arg2 : tensor<256xf16>) -> (tensor<100xf32>) {
  %empty0 = tensor.empty() : tensor<256xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg2 : tensor<f16>, tensor<256xf16>) outs(%empty0 : tensor<256xf32>) {
  ^bb0(%in: f16, %in0 : f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.extf %in0 : f16 to f32
    %2 = arith.addf %0, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<256xf32>
  %empty1 = tensor.empty() : tensor<100xf32>
  %gather = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg1: tensor<100xindex>) outs(%empty1 : tensor<100xf32>) {
  ^bb0(%in: index, %out: f32):
    %1 = tensor.extract %0[%in] : tensor<256xf32>
    linalg.yield %1 : f32
  } -> tensor<100xf32>
  util.return %gather : tensor<100xf32>
}
// CHECK-LABEL: util.func public @gather_0d_producer(
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9]+]]: tensor
//       CHECK:   %[[GATHER:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[ARG1]] : tensor<100xindex>
//  CHECK-NEXT:     ^bb0(%[[IN:.+]]: index
//   CHECK-DAG:     %[[EXTRACT0:.+]] = tensor.extract %[[ARG0]][]
//   CHECK-DAG:     %[[EXTRACT1:.+]] = tensor.extract %[[ARG2]][%[[IN]]]
//       CHECK:   return %[[GATHER]]

// -----

util.func public @gather_replace_linalg_index(%arg0 : tensor<256x256xf16>, %arg1 : tensor<100xindex>) -> (tensor<100xf32>) {
  %empty0 = tensor.empty() : tensor<256x256xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<256x256xf16>) outs(%empty0 : tensor<256x256xf32>) {
  ^bb0(%in: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = linalg.index 1 : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.uitofp %2 : i32 to f32
    %4 = arith.addf %0, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<256x256xf32>
  %empty1 = tensor.empty() : tensor<100xf32>
  %gather = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg1: tensor<100xindex>) outs(%empty1 : tensor<100xf32>) {
  ^bb0(%in: index, %out: f32):
    %cst0 = arith.constant 0 : index
    %1 = tensor.extract %0[%cst0, %in] : tensor<256x256xf32>
    linalg.yield %1 : f32
  } -> tensor<100xf32>
  util.return %gather : tensor<100xf32>
}
// CHECK-LABEL: util.func public @gather_replace_linalg_index(
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//       CHECK:   %[[GATHER:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[ARG1]] : tensor<100xindex>
//  CHECK-NEXT:     ^bb0(%[[IN:.+]]: index
//       CHECK:     arith.index_cast %[[IN]]
//       CHECK:   return %[[GATHER]]

// -----

util.func public @gather_replace_linalg_index_transpose(%arg0 : tensor<256x256xf16>, %arg1 : tensor<100xindex>, %arg2 : index) -> (tensor<100xf32>) {
  %empty0 = tensor.empty() : tensor<256x256xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<256x256xf16>) outs(%empty0 : tensor<256x256xf32>) {
  ^bb0(%in: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = linalg.index 1 : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.uitofp %2 : i32 to f32
    %4 = arith.addf %0, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<256x256xf32>
  %empty1 = tensor.empty() : tensor<100xf32>
  %gather = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg1: tensor<100xindex>) outs(%empty1 : tensor<100xf32>) {
  ^bb0(%in: index, %out: f32):
    %1 = tensor.extract %0[%arg2, %in] : tensor<256x256xf32>
    linalg.yield %1 : f32
  } -> tensor<100xf32>
  util.return %gather : tensor<100xf32>
}
// CHECK-LABEL: util.func public @gather_replace_linalg_index_transpose(
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9]+]]: index
//       CHECK:   %[[GATHER:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[ARG1]] : tensor<100xindex>
//       CHECK:     arith.index_cast %[[ARG2]]
//       CHECK:   return %[[GATHER]]
