// RUN: iree-opt --iree-flow-fusion-preprocessing --split-input-file %s | FileCheck %s

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1)>
//      CHECK: util.func public @interchange
//      CHECK:   linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
util.func public @interchange(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
  %0 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>],
    iterator_types = ["reduction", "parallel", "parallel", "parallel"]}
  ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  outs(%arg2 : tensor<?x?x?xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %m = arith.mulf %arg3, %arg4 : f32
    %a = arith.addf %arg5, %m : f32
    linalg.yield %a : f32
  } -> tensor<?x?x?xf32>
  util.return %0 : tensor<?x?x?xf32>
}

// -----

util.func public @fold_insert_slices(%source : tensor<?x?xf32>,
    %dest0 : tensor<?x?xf32>, %dest1 : tensor<?x?xf32>, %val: f32,
    %o1 : index, %o2 : index, %o3 : index, %o4 : index,
    %s1 : index, %s2 : index, %s3 : index, %s4 : index) -> tensor<?x?xf32> {
  %0 = linalg.fill ins(%val : f32) outs(%dest0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = tensor.insert_slice %source into %0[%o1, %o2] [%s1, %s2] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  %2 = linalg.fill ins(%val : f32) outs(%dest1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = tensor.insert_slice %1 into %2[%o3, %o4] [%s3, %s4] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %3 : tensor<?x?xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//      CHECK: func public @fold_insert_slices
// CHECK-SAME:     %[[SOURCE:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[DEST0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[DEST1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[CST:.+]]: f32
// CHECK-SAME:     %[[OFFSET0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[OFFSET1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[OFFSET2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[OFFSET3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SIZE0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SIZE1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SIZE2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SIZE3:[a-zA-Z0-9]+]]: index
//      CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[DEST1]] :
//  CHECK-DAG:   %[[NEW_OFFSET0:.+]] = affine.apply #[[MAP]]()[%[[OFFSET0]], %[[OFFSET2]]]
//  CHECK-DAG:   %[[NEW_OFFSET1:.+]] = affine.apply #[[MAP]]()[%[[OFFSET1]], %[[OFFSET3]]]
//      CHECK:   %[[RETURN:.+]] = tensor.insert_slice %[[SOURCE]] into %[[FILL]]
// CHECK-SAME:       [%[[NEW_OFFSET0]], %[[NEW_OFFSET1]]] [%[[SIZE0]], %[[SIZE1]]]
//      CHECK:   util.return %[[RETURN]]


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

util.func public @fuse_attention_expand_transpose(
  %arg0: tensor<?x?x?xf16>, %arg1 : tensor<?x?x?xf16>, %arg2 : tensor<?x?x?xf16>, %arg3 : f16) -> tensor<2x?x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf16>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf16>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf16>
  %d3 = tensor.dim %arg1, %c1 : tensor<?x?x?xf16>
  %d4 = tensor.dim %arg2, %c2 : tensor<?x?x?xf16>
  %empty = tensor.empty(%d0, %d1, %d4) : tensor<?x?x?xf16>
  %attention = iree_linalg_ext.attention {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
    ins(%arg0, %arg1, %arg2, %arg3 : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f16)
    outs(%empty : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %split = arith.divsi %d0, %c2 : index
  %expanded = tensor.expand_shape %attention [[0, 1], [2], [3]] output_shape[2, %split, %d1, %d4]
      : tensor<?x?x?xf16> into tensor<2x?x?x?xf16>
  %empty2 = tensor.empty(%d1, %split, %d4) : tensor<2x?x?x?xf16>
  %transpose = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<2x?x?x?xf16>) outs(%empty2 : tensor<2x?x?x?xf16>) {
    ^bb0(%b0 : f16, %b1 : f16):
      linalg.yield %b0 : f16
  } -> tensor<2x?x?x?xf16>
  util.return %transpose : tensor<2x?x?x?xf16>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d5)>
//      CHECK: func public @fuse_attention_expand_transpose(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D4:.+]] = tensor.dim %[[ARG2]], %[[C2]]
//  CHECK-DAG:   %[[D_SPLIT:.+]] = arith.divsi %[[D0]], %[[C2]]
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[D1]], %[[D_SPLIT]], %[[D4]]) : tensor<2x?x?x?xf16>
//  CHECK-DAG:   %[[D_SPLIT2:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//  CHECK-DAG:   %[[D3:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[QUERY:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[D_SPLIT2]], %[[D1]], %[[D2]]{{\]}}
//  CHECK-DAG:   %[[KEY:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[D_SPLIT2]], %[[D3]], %[[D2]]{{\]}}
//  CHECK-DAG:   %[[CACHE:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, %[[D_SPLIT2]], %[[D3]], %[[D4]]{{\]}}
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
// CHECK-SAME:       indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP4]]]
// CHECK-SAME:       ins(%[[QUERY]], %[[KEY]], %[[CACHE]], %[[ARG3]] :
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   util.return %[[ATTENTION]]

// -----

util.func public @fuse_attention_expand_transpose_static(
      %arg0 : tensor<20x4096x16xf16>, %arg1 : tensor<20x1024x16xf16>,
      %arg2 : tensor<20x1024x64xf16>, %arg3 : f16) -> tensor<2x4096x10x64xf16> {
  %empty = tensor.empty() : tensor<20x4096x64xf16>
  %attention = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
      ins(%arg0, %arg1, %arg2, %arg3 : tensor<20x4096x16xf16>, tensor<20x1024x16xf16>, tensor<20x1024x64xf16>, f16)
      outs(%empty: tensor<20x4096x64xf16>) -> tensor<20x4096x64xf16>
  %expanded = tensor.expand_shape %attention [[0, 1], [2], [3]]
      output_shape [2, 10, 4096, 64] : tensor<20x4096x64xf16> into tensor<2x10x4096x64xf16>
  %empty2 = tensor.empty() : tensor<2x4096x10x64xf16>
  %transpose = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<2x10x4096x64xf16>) outs(%empty2 : tensor<2x4096x10x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
  } -> tensor<2x4096x10x64xf16>
  util.return %transpose : tensor<2x4096x10x64xf16>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d5)>
//      CHECK: func public @fuse_attention_expand_transpose_static(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<20x4096x16xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<20x1024x16xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<20x1024x64xf16>
// CHECK-SAME:     %[[ARG3:.+]]: f16)
//      CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2x4096x10x64xf16>
//  CHECK-DAG:   %[[QUERY:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 4096, 16]
//  CHECK-DAG:   %[[KEY:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 1024, 16]
//  CHECK-DAG:   %[[CACHE:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [2, 10, 1024, 64]
//      CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK-SAME:       ins(%[[QUERY]], %[[KEY]], %[[CACHE]], %[[ARG3]] :
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   util.return %[[ATTENTION]]
