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
