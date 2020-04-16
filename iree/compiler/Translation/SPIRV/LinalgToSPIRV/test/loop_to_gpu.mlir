// RUN: iree-opt -iree-convert-to-gpu -canonicalize -split-input-file %s | IreeFileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
#map1 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func @pw_add(%arg0: memref<4x8xi32>, %arg1: memref<4x8xi32>,
               %arg2: memref<4x8xi32>)
  attributes {iree.dispatch_fn_name = "pw_add"} {
    %c32 = constant 32 : index
    %c0 = constant 0 : index
    %c4 = constant 4 : index
    %c8 = constant 8 : index
    %c1 = constant 1 : index
    loop.parallel (%arg3, %arg4) = (%c0, %c0) to (%c4, %c8) step (%c4, %c32) {
      %0 = affine.min #map0(%c4, %c4, %arg3)
      %1 = affine.min #map0(%c32, %c8, %arg4)
      %2 = subview %arg0[%arg3, %arg4] [%0, %1] [%c1, %c1]
             : memref<4x8xi32> to memref<?x?xi32, #map1>
      %3 = affine.min #map0(%c4, %c4, %arg3)
      %4 = affine.min #map0(%c32, %c8, %arg4)
      %5 = subview %arg1[%arg3, %arg4] [%3, %4] [%c1, %c1]
             : memref<4x8xi32> to memref<?x?xi32, #map1>
      %6 = affine.min #map0(%c4, %c4, %arg3)
      %7 = affine.min #map0(%c32, %c8, %arg4)
      %8 = subview %arg2[%arg3, %arg4] [%6, %7] [%c1, %c1]
             : memref<4x8xi32> to memref<?x?xi32, #map1>
      linalg.generic
        {args_in = 2 : i64, args_out = 1 : i64,
         indexing_maps = [#map2, #map2, #map2],
         iterator_types = ["parallel", "parallel"]}
      {__internal_linalg_transform__ = "workitem"} %2, %5, %8 {
      ^bb0(%arg5: i32, %arg6: i32, %arg7: i32): // no predecessors
        %9 = addi %arg5, %arg6 : i32
        linalg.yield %9 : i32
      } : memref<?x?xi32, #map1>, memref<?x?xi32, #map1>, memref<?x?xi32, #map1>
      loop.yield
    }
    return
  }
}
//   CHECK-DAG:   %[[STEPY:.*]] = constant 4 : index
//   CHECK-DAG:   %[[STEPX:.*]] = constant 32 : index
//   CHECK-DAG:   %[[BIDX:.*]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[NBLOCKSX:.*]] = "gpu.grid_dim"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.*]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[NBLOCKSY:.*]] = "gpu.grid_dim"() {dimension = "y"}
//       CHECK:   %[[NEWLBY:.*]] = muli %[[BIDY]], %[[STEPY]]
//       CHECK:   %[[NEWSTEPY:.*]] = muli %[[NBLOCKSY]], %[[STEPY]]
//       CHECK:   %[[NEWLBX:.*]] = muli %[[BIDX]], %[[STEPX]]
//       CHECK:   %[[NEWSTEPX:.*]] = muli %[[NBLOCKSX]], %[[STEPX]]
//       CHECK:   loop.for %{{.*}} = %[[NEWLBY]] to %{{.*}} step %[[NEWSTEPY]]
//       CHECK:     loop.for %{{.*}} = %[[NEWLBX]] to %{{.*}} step %[[NEWSTEPX]]
//   CHECK-DAG:       %[[TIDX:.*]] = "gpu.thread_id"() {dimension = "x"}
//   CHECK-DAG:       %[[NTHREADSX:.*]] = "gpu.block_dim"() {dimension = "x"}
//   CHECK-DAG:       %[[TIDY:.*]] = "gpu.thread_id"() {dimension = "y"}
//   CHECK-DAG:       %[[NTHREADSY:.*]] = "gpu.block_dim"() {dimension = "y"}
//       CHECK:       loop.for %{{.*}} = %[[TIDY]] to %{{.*}} step %[[NTHREADSY]]
//       CHECK:         loop.for %{{.*}} = %[[TIDX]] to %{{.*}} step %[[NTHREADSX]]

// -----

module {
  func @reduce_sum(%arg0: memref<4xf32>, %arg1: memref<f32>, %arg2: memref<f32>)
   attributes {iree.dispatch_fn_name = "reduce_sum"} {
    linalg.indexed_generic
      {args_in = 2 : i64, args_out = 1 : i64,
       indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>,
                        affine_map<(d0) -> ()>],
       iterator_types = ["reduction"]} %arg0, %arg1, %arg2 {
    ^bb0(%arg3: index, %arg4: f32, %arg5: f32, %arg6: f32):   // no predecessors
      %c0 = constant 0 : index
      %cst = constant true
      %0 = cmpi "eq", %arg3, %c0 : index
      %1 = and %cst, %0 : i1
      %2 = select %1, %arg5, %arg6 : f32
      %3 = addf %arg4, %2 : f32
      linalg.yield %3 : f32
    }: memref<4xf32>, memref<f32>, memref<f32>
    return
  }
}
// CHECK-DAG: %[[C0:.*]] = constant 0 : index
// CHECK-DAG: %[[C4:.*]] = constant 4 : index
// CHECK-DAG: %[[C1:.*]] = constant 1 : index
//     CHECK:   loop.for %{{.*}} = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-NOT:   loop

// -----

#map0 = affine_map<(d0)[s0] -> (2, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (32, -d0 + s0)>
#map2 = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func @parallel_4D(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) attributes {iree.dispatch_fn_name = "parallel_4D", spv.entry_point_abi = {local_size = dense<[32, 2, 2]> : vector<3xi32>}} {
    %c2 = constant 2 : index
    %c32 = constant 32 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = dim %arg0, 0 : memref<?x?x?x?xf32>
    %1 = dim %arg0, 1 : memref<?x?x?x?xf32>
    %2 = dim %arg0, 2 : memref<?x?x?x?xf32>
    %3 = dim %arg0, 3 : memref<?x?x?x?xf32>
    loop.parallel (%arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0) to (%0, %1, %2, %3) step (%c2, %c2, %c2, %c32) {
      %12 = affine.min #map0(%arg3)[%0]
      %13 = affine.min #map0(%arg4)[%1]
      %14 = affine.min #map0(%arg5)[%2]
      %15 = affine.min #map1(%arg6)[%3]
      %16 = subview %arg0[%arg3, %arg4, %arg5, %c0] [%12, %13, %14, %15] [%c1, %c1, %c1, %c1] : memref<?x?x?x?xf32> to memref<?x?x?x?xf32, #map2>
      %17 = subview %arg1[%arg3, %arg4, %arg5, %c0] [%12, %13, %14, %15] [%c1, %c1, %c1, %c1] : memref<?x?x?x?xf32> to memref<?x?x?x?xf32, #map2>
      %18 = subview %arg2[%arg3, %arg4, %arg5, %c0] [%12, %13, %14, %15] [%c1, %c1, %c1, %c1] : memref<?x?x?x?xf32> to memref<?x?x?x?xf32, #map2>
      linalg.generic {args_in = 2 : i64, args_out = 1 : i64,
        indexing_maps = [#map3, #map3, #map3],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      {__internal_linalg_transform__ = "workitem"}
      %16, %17, %18
      {
        ^bb0(%arg7: f32, %arg8: f32, %arg9: f32): // no predecessors
          %19 = addf %arg7, %arg8 : f32
          linalg.yield %19 : f32
      } : memref<?x?x?x?xf32, #map2>, memref<?x?x?x?xf32, #map2>, memref<?x?x?x?xf32, #map2>
      loop.yield
    }
    return
  }
}

// CHECK-DAG: %[[C2:.*]] = constant 2 : index
// CHECK-DAG: %[[C32:.*]] = constant 32 : index
// CHECK-DAG: %[[C0:.*]] = constant 0 : index
// CHECK-DAG: %[[C1:.*]] = constant 1 : index
// CHECK-DAG: %[[SERIALDIMOUTER:.*]] = dim %{{.*}}, 3
// CHECK-DAG: %[[BIDX:.*]] = "gpu.block_id"() {dimension = "x"} : () -> index
// CHECK-DAG: %[[NBLOCKSX:.*]] = "gpu.grid_dim"() {dimension = "x"} : () -> index
// CHECK-DAG: %[[BIDY:.*]] = "gpu.block_id"() {dimension = "y"} : () -> index
// CHECK-DAG: %[[NBLOCKSY:.*]] = "gpu.grid_dim"() {dimension = "y"} : () -> index
// CHECK-DAG: %[[BIDZ:.*]] = "gpu.block_id"() {dimension = "z"} : () -> index
// CHECK-DAG: %[[NBLOCKSZ:.*]] = "gpu.grid_dim"() {dimension = "z"} : () -> index
// CHECK-DAG: %[[LB0:.*]] = muli %[[BIDZ]], %[[C2]]
// CHECK-DAG: %[[STEP0:.*]] = muli %[[NBLOCKSZ]], %[[C2]]
// CHECK-DAG: %[[LB1:.*]] = muli %[[BIDY]], %[[C2]]
// CHECK-DAG: %[[STEP1:.*]] = muli %[[NBLOCKSY]], %[[C2]]
// CHECK-DAG: %[[LB2:.*]] = muli %[[BIDX]], %[[C2]]
// CHECK-DAG: %[[STEP2:.*]] = muli %[[NBLOCKSX]], %[[C2]]
//     CHECK: loop.for %{{.*}} = %[[LB0]] to %{{.*}} step %[[STEP0]]
//     CHECK:   loop.for %{{.*}} = %[[LB1]] to %{{.*}} step %[[STEP1]]
//     CHECK:     loop.for %{{.*}} = %[[LB2]] to %{{.*}} step %[[STEP2]]
//     CHECK:       loop.for %{{.*}} = %[[C0]] to %[[SERIALDIMOUTER]] step %[[C32]]
// CHECK-DAG:         %[[TIDX:.*]] = "gpu.thread_id"() {dimension = "x"} : () -> index
// CHECK-DAG:         %[[NTHREADSX:.*]] = "gpu.block_dim"() {dimension = "x"} : () -> index
// CHECK-DAG:         %[[TIDY:.*]] = "gpu.thread_id"() {dimension = "y"} : () -> index
// CHECK-DAG:         %[[NTHREADSY:.*]] = "gpu.block_dim"() {dimension = "y"} : () -> index
// CHECK-DAG:         %[[TIDZ:.*]] = "gpu.thread_id"() {dimension = "z"} : () -> index
// CHECK-DAG:         %[[NTHREADSZ:.*]] = "gpu.block_dim"() {dimension = "z"} : () -> index
//     CHECK:         loop.for %{{.*}} = %[[TIDZ]] to %{{.*}} step %[[NTHREADSZ]]
//     CHECK:           loop.for %{{.*}} = %[[TIDY]] to %{{.*}} step %[[NTHREADSY]]
//     CHECK:             loop.for %{{.*}} = %[[TIDX]] to %{{.*}} step %[[NTHREADSX]]
//     CHECK:               loop.for %{{.*}} = %[[C0]] to %{{.*}} step %[[C1]]
