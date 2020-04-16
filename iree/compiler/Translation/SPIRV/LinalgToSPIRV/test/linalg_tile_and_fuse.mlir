// RUN: iree-opt -split-input-file -iree-linalg-tile-and-fuse %s | IreeFileCheck %s

module {
  // CHECK-LABEL: func @tile_only
  //  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<4x8xi32>
  //  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<4x8xi32>
  //  CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<4x8xi32>
  //       CHECK: loop.parallel
  //       CHECK:   %[[VIEW0:.*]] = subview %[[ARG0]]
  //       CHECK:   %[[VIEW1:.*]] = subview %[[ARG1]]
  //       CHECK:   %[[VIEW2:.*]] = subview %[[ARG2]]
  //       CHECK:   linalg.generic
  //  CHECK-SAME:     %[[VIEW0]]
  //  CHECK-SAME:     %[[VIEW1]]
  //  CHECK-SAME:     %[[VIEW2]]
  func @tile_only(%arg0: memref<4x8xi32>, %arg1: memref<4x8xi32>,
                  %arg2: memref<4x8xi32>)
  attributes
    {iree.dispatch_fn_name = "tile_only"} {
    linalg.generic
      {args_in = 2 : i64, args_out = 1 : i64,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} %arg0, %arg1, %arg2 {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %0 = addi %arg3, %arg4 : i32
      linalg.yield %0 : i32
    }: memref<4x8xi32>, memref<4x8xi32>, memref<4x8xi32>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @tile_and_fuse
  //  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?xf32>
  //  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?xf32>
  //  CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?xf32>
  //  CHECK-SAME: %[[ARG3:[a-zA-Z0-9_]*]]: memref<?x?xf32>
  //       CHECK: loop.parallel
  //   CHECK-DAG:   %[[VIEW0:.*]] = subview %[[ARG0]]
  //   CHECK-DAG:   %[[VIEW1:.*]] = subview %[[ARG1]]
  //   CHECK-DAG:   %[[VIEW2READ:.*]] = subview %[[ARG2]]
  //   CHECK-DAG:   %[[VIEW2WRITE:.*]] = subview %[[ARG2]]
  //   CHECK-DAG:   %[[VIEW3:.*]] = subview %[[ARG3]]
  //       CHECK:   linalg.generic
  //  CHECK-SAME:     %[[VIEW0]]
  //  CHECK-SAME:     %[[VIEW1]]
  //  CHECK-SAME:     %[[VIEW2WRITE]]
  //       CHECK:   linalg.generic
  //  CHECK-SAME:     %[[VIEW2READ]]
  //  CHECK-SAME:     %[[VIEW3]]
  func @tile_and_fuse(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                      %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>)
  attributes
    {iree.dispatch_fn_name = "tile_and_fuse"} {
    linalg.generic
      {args_in = 2 : i64, args_out = 1 : i64,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} %arg0, %arg1, %arg2 {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %0 = addf %arg4, %arg5 : f32
      linalg.yield %0 : f32
    }: memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
    linalg.generic
      {args_in = 1 : i64, args_out = 1 : i64,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} %arg2, %arg3 {
    ^bb0(%arg7: f32, %arg8: f32):
      %1 = mulf %arg7, %arg7 : f32
      linalg.yield %1 : f32
    }: memref<?x?xf32>, memref<?x?xf32>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @conv_padding
  //  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32>
  //  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32>
  //  CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32>
  //       CHECK: loop.parallel (%{{.*}})
  //       CHECK:   %[[VIEW1:.*]] = subview %[[ARG1]]
  //       CHECK:   %[[VIEW2:.*]] = subview %[[ARG2]]
  //       CHECK:   linalg.conv
  //  CHECK-SAME:     %[[VIEW1]]
  //  CHECK-SAME:     %[[VIEW2]]
  func @conv_padding(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>,
                     %arg2 : memref<?x?x?x?xf32>)
    attributes
      {iree.dispatch_fn_name = "conv_padding"} {
    linalg.conv(%arg0, %arg1, %arg2)
      {dilations = [1, 1],
       padding = dense<[[1, 1], [0, 1]]> : tensor<2x2xi64>, strides = [1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @conv_no_padding
  //  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32>
  //  CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32>
  //  CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32>
  //       CHECK: loop.parallel (%{{.*}}, %{{.*}}, %{{.*}})
  //       CHECK:   %[[VIEW1:.*]] = subview %[[ARG1]]
  //       CHECK:   %[[VIEW2:.*]] = subview %[[ARG2]]
  //       CHECK:   linalg.conv
  //  CHECK-SAME:     %[[VIEW1]]
  //  CHECK-SAME:     %[[VIEW2]]
  func @conv_no_padding(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>,
                        %arg2 : memref<?x?x?x?xf32>)
    attributes
      {iree.dispatch_fn_name = "conv_no_padding"} {
    linalg.conv(%arg0, %arg1, %arg2) {dilations = [1, 1], strides = [1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK-LABEL: func @parallel_4D
  //       CHECK: loop.parallel (%{{.*}}, %{{.*}}, %{{.*}})
  func @parallel_4D(%arg0: memref<?x?x?x?xf32>,
                    %arg1 : memref<?x?x?x?xf32>,
                    %arg2 : memref<?x?x?x?xf32>)
  attributes {iree.dispatch_fn_name = "parallel_4D"} {
    linalg.generic
      {args_in = 2 : i64, args_out = 1 : i64,
       indexing_maps = [#map0, #map0, #map0],
       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      %arg0, %arg1, %arg2 {
    ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
      %0 = addf %arg3, %arg4 : f32
      linalg.yield %0 : f32
    } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
}
