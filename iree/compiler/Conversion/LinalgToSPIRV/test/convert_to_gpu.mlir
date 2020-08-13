// RUN: iree-opt -iree-codegen-convert-to-gpu -canonicalize -cse -split-input-file %s | IreeFileCheck %s

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
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
// CHECK-LABEL: func @parallel_4D
//  CHECK-SAME:   local_size = dense<[32, 1, 1]>
//  CHECK-SAME:   vkspv.workgroup_count_from_result_shape = 1
//   CHECK-DAG:     %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:     %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:     %[[C3:.+]] = constant 3 : index
//   CHECK-DAG:     %[[UB0:.+]] = dim %{{.+}}, %[[C0]]
//   CHECK-DAG:     %[[UB1:.+]] = dim %{{.+}}, %[[C1]]
//   CHECK-DAG:     %[[UB2:.+]] = dim %{{.+}}, %[[C2]]
//   CHECK-DAG:     %[[UB3:.+]] = dim %{{.+}}, %[[C3]]
//       CHECK:     %[[T4:.+]] = muli %[[UB3]], %[[UB2]]
//       CHECK:     %[[T5:.+]] = muli %[[T4]], %[[UB1]]
//       CHECK:     %[[UB:.+]] = muli %[[T5]], %[[UB0]]
//   CHECK-DAG:     %[[BID:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:     %[[BDIM:.+]] = "gpu.block_dim"() {dimension = "x"}
//   CHECK-DAG:     %[[TID:.+]] = "gpu.thread_id"() {dimension = "x"}
//       CHECK:     %[[BOFFSET:.+]] = muli %[[BID]], %[[BDIM]]
//       CHECK:     %[[IV:.+]] = addi %[[BOFFSET]], %[[TID]]
//       CHECK:     %[[COND:.+]] = cmpi "slt", %[[IV]], %[[UB]]
//       CHECK:     scf.if %[[COND]]
//       CHECK:       %[[IV0:.+]] = divi_signed %[[IV]], %[[T5]]
//       CHECK:       %[[T14:.+]] = remi_signed %[[IV]], %[[T5]]
//       CHECK:       %[[IV1:.+]] = divi_signed %[[T14]], %[[T4]]
//       CHECK:       %[[T16:.+]] = remi_signed %[[T14]], %[[T4]]
//       CHECK:       %[[IV2:.+]] = divi_signed %[[T16]], %[[UB3]]
//       CHECK:       %[[IV3:.+]] = remi_signed %[[T16]], %[[UB3]]
//       CHECK:       load %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:       load %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:       store %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]


// -----

#map0 = affine_map<() -> ()>
#accesses = [#map0, #map0, #map0]
#trait = {
  args_in = 2 : i64,
  args_out = 1 : i64,
  indexing_maps = #accesses,
  iterator_types = []
}

module attributes {
  spv.target_env =
    #spv.target_env<#spv.vce<v1.3,
    [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @scalar_add(%arg0 : memref<f32>, %arg1 : memref<f32>,
                   %arg2 : memref<f32>)
  {
    linalg.generic #trait %arg0, %arg1, %arg2 {
    ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
      %0 = addf %arg3, %arg4 : f32
      linalg.yield %0 : f32
     } : memref<f32>, memref<f32>, memref<f32>
     return
  }
}
// CHECK-LABEL: func @scalar_add
//  CHECK-SAME:   local_size = dense<1> : vector<3xi32>
//  CHECK-SAME:   vkspv.workgroup_count_from_result_shape = 1
//  CHECK-NEXT:     load
//  CHECK-NEXT:     load
//  CHECK-NEXT:     addf
//  CHECK-NEXT:     store
//  CHECK-NEXT:     return

// -----

module {
  func @reduce_sum(%arg0: memref<?x?x?xf32>, %arg1: memref<f32>, %arg2: memref<?xf32>)
   attributes {iree.dispatch_fn_name = "reduce_sum"} {
    linalg.indexed_generic
      {args_in = 2 : i64, args_out = 1 : i64,
       indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0)>],
       iterator_types = ["parallel", "parallel", "reduction"]} %arg0, %arg1, %arg2 {
    ^bb0(%arg3: index, %arg4: index, %arg5: index,
         %arg6: f32, %arg7: f32, %arg8: f32):   // no predecessors
      %c0 = constant 0 : index
      %cst = constant true
      %0 = cmpi "eq", %arg5, %c0 : index
      %1 = and %cst, %0 : i1
      %2 = select %1, %arg7, %arg8 : f32
      %3 = addf %arg6, %2 : f32
      linalg.yield %3 : f32
    }: memref<?x?x?xf32>, memref<f32>, memref<?xf32>
    return
  }
}

// CHECK-LABEL: func @reduce_sum
//  CHECK-SAME:   local_size = dense<[32, 1, 1]> : vector<3xi32>
//  CHECK-SAME:   vkspv.workgroup_count_from_result_shape = 1
//   CHECK-DAG:     %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:     %[[C2:.+]] = constant 2 : index
//       CHECK:     %[[UB0:.+]] = dim %{{.+}}, %[[C0]]
//       CHECK:     %[[UB1:.+]] = dim %{{.+}}, %[[C1]]
//       CHECK:     %[[UB2:.+]] = dim %{{.+}}, %[[C2]]
//       CHECK:     %[[UB:.+]] = muli %[[UB1]], %[[UB0]]
//       CHECK:     %[[COND:.+]] = cmpi "slt", %{{.+}}, %[[UB]]
//       CHECK:     scf.if %[[COND]]
//       CHECK:       %[[IV0:.+]] = divi_signed %{{.+}}, %[[UB1]]
//       CHECK:       %[[IV1:.+]] = remi_signed %{{.+}}, %[[UB1]]
//       CHECK:       scf.for %[[IV:.+]] = %{{.+}} to %[[UB2]]
//       CHECK:         %[[ISZERO:.+]] = cmpi "eq", %[[IV]], %[[C0]]

// -----

#map0 = affine_map<(d0)[s0] -> (8, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map2 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) attributes {spv.entry_point_abi = {local_size = dense<[8, 8, 1]> : vector<3xi32>}} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c4 = constant 4 : index
    %c8 = constant 8 : index
    %0 = dim %arg0, %c0 : memref<?x?xf32>
    %1 = dim %arg0, %c1 : memref<?x?xf32>
    %2 = dim %arg1, %c1 : memref<?x?xf32>
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%0, %2) step (%c8, %c8) {
      scf.for %arg5 = %c0 to %1 step %c4 {
        %3 = affine.min #map0(%arg3)[%0]
        %4 = affine.min #map1(%arg5)[%1]
        %5 = subview %arg0[%arg3, %arg5] [%3, %4] [1, 1]  : memref<?x?xf32> to memref<?x?xf32, #map2>
        %6 = dim %arg1, %c0 : memref<?x?xf32>
        %7 = affine.min #map1(%arg5)[%6]
        %8 = affine.min #map0(%arg4)[%2]
        %9 = subview %arg1[%arg5, %arg4] [%7, %8] [1, 1]  : memref<?x?xf32> to memref<?x?xf32, #map2>
        %10 = dim %arg2, %c0 : memref<?x?xf32>
        %11 = affine.min #map0(%arg3)[%10]
        %12 = dim %arg2, %c1 : memref<?x?xf32>
        %13 = affine.min #map0(%arg4)[%12]
        %14 = subview %arg2[%arg3, %arg4] [%11, %13] [1, 1]  : memref<?x?xf32> to memref<?x?xf32, #map2>
        linalg.matmul %5, %9, %14 {__internal_linalg_transform__ = "workgroup_numprocs_ge_numiters"} : (memref<?x?xf32, #map2>, memref<?x?xf32, #map2>, memref<?x?xf32, #map2>)
      }
      scf.yield
    }
    return
  }
}

// CHECK-LABEL: func @matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C8:.+]] = constant 8 : index
//   CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[UB0:.+]] = dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[UB1:.+]] = dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[UB2:.+]] = dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//       CHECK:   %[[BOFFSETY:.+]] = muli %[[BIDY]], %[[C8]]
//       CHECK:   %[[BOFFSETX:.+]] = muli %[[BIDX]], %[[C8]]
//       CHECK:   scf.for %[[KOFFSET:.+]] = %[[C0]] to %[[UB2]] step %[[C4]]
//       CHECK:     %[[VIEWUB0:.+]] = affine.min #{{.+}}()[%[[UB0]], %[[BOFFSETY]]]
//       CHECK:     %[[VIEWUB2:.+]] = affine.min #{{.+}}(%[[KOFFSET]])[%[[UB2]]]
//       CHECK:     %[[SV0:.+]] = subview %[[ARG0]][%[[BOFFSETY]], %[[KOFFSET]]]
//       CHECK:     %[[VIEWUB1:.+]] = affine.min #{{.+}}()[%[[UB1]], %[[BOFFSETX]]]
//       CHECK:     %[[SV1:.+]] = subview %[[ARG1]][%[[KOFFSET]], %[[BOFFSETX]]]
//       CHECK:     %[[SV2:.+]] = subview %[[ARG2]][%[[BOFFSETY]], %[[BOFFSETX]]]
//   CHECK-DAG:     %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//   CHECK-DAG:     %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//       CHECK:     %[[INBOUNDY:.+]] = cmpi "slt", %[[TIDY]], %[[VIEWUB0]]
//       CHECK:     %[[INBOUNDX:.+]] = cmpi "slt", %[[TIDX]], %[[VIEWUB1]]
//       CHECK:     %[[COND:.+]] = and %[[INBOUNDY]], %[[INBOUNDX]]
//       CHECK:     scf.if %[[COND]]
//       CHECK:       scf.for %{{.+}} = %[[C0]] to %[[VIEWUB2]] step %[[C1]]
//   CHECK-NOT:         linalg.matmul

// -----

#map0 = affine_map<(d0)[s0] -> (1, -d0 + s0)>
#map1 = affine_map<(d0)[s0, s1] -> (s0 + 4, -d0 + s1)>
#map2 = affine_map<(d0)[s0, s1] -> (s0 + 32, -d0 + s1)>
#map3 = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
#map4 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map5 = affine_map<(d0)[s0] -> (32, -d0 + s0)>

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @conv_no_padding(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}} {
    %c4 = constant 4 : index
    %c32 = constant 32 : index
    %c2 = constant 2 : index
    %c0 = constant 0 : index
    %c3 = constant 3 : index
    %c1 = constant 1 : index
    %0 = dim %arg1, %c0 : memref<?x?x?x?xf32>
    %1 = dim %arg1, %c1 : memref<?x?x?x?xf32>
    %2 = dim %arg1, %c2 : memref<?x?x?x?xf32>
    %3 = dim %arg2, %c1 : memref<?x?x?x?xf32>
    %4 = dim %arg2, %c2 : memref<?x?x?x?xf32>
    scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%0, %3, %4) step (%c1, %c4, %c32) {
      %5 = affine.min #map0(%arg3)[%0]
      %6 = affine.min #map1(%arg4)[%1, %1]
      %7 = affine.min #map2(%arg5)[%2, %2]
      %8 = dim %arg1, %c3 : memref<?x?x?x?xf32>
      %9 = subview %arg1[%arg3, %arg4, %arg5, 0] [%5, %6, %7, %8] [1, 1, 1, 1]  : memref<?x?x?x?xf32> to memref<?x?x?x?xf32, #map3>
      %10 = dim %arg2, %c0 : memref<?x?x?x?xf32>
      %11 = affine.min #map0(%arg3)[%10]
      %12 = affine.min #map4(%arg4)[%3]
      %13 = affine.min #map5(%arg5)[%4]
      %14 = dim %arg2, %c3 : memref<?x?x?x?xf32>
      %15 = subview %arg2[%arg3, %arg4, %arg5, 0] [%11, %12, %13, %14] [1, 1, 1, 1]  : memref<?x?x?x?xf32> to memref<?x?x?x?xf32, #map3>
      linalg.conv(%arg0, %9, %15) {__internal_linalg_transform__ = "workgroup", dilations = [1, 1], strides = [1, 1]} : memref<?x?x?x?xf32>, memref<?x?x?x?xf32, #map3>, memref<?x?x?x?xf32, #map3>
      scf.yield
    }
    return
  }
}

// CHECK-LABEL: func @conv_no_padding
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
//   CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//   CHECK-DAG:   %[[C32:.+]] = constant 32 : index
//   CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[UB0:.+]] = dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[UB1:.+]] = dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[UB2:.+]] = dim %[[ARG1]], %[[C2]]
//   CHECK-DAG:   %[[UB3:.+]] = dim %[[ARG2]], %[[C1]]
//   CHECK-DAG:   %[[UB4:.+]] = dim %[[ARG2]], %[[C2]]
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[NBLOCKSX:.+]] = "gpu.grid_dim"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[NBLOCKSY:.+]] = "gpu.grid_dim"() {dimension = "y"}
//   CHECK-DAG:   %[[BIDZ:.+]] = "gpu.block_id"() {dimension = "z"}
//   CHECK-DAG:   %[[NBLOCKSZ:.+]] = "gpu.grid_dim"() {dimension = "z"}
//       CHECK:   %[[BOFFSETY:.+]] = muli %[[BIDY]], %[[C4]]
//       CHECK:   %[[BSTEPY:.+]] = muli %[[NBLOCKSY]], %[[C4]]
//       CHECK:   %[[BOFFSETX:.+]] = muli %[[BIDX]], %[[C32]]
//       CHECK:   %[[BSTEPX:.+]] = muli %[[NBLOCKSX]], %[[C32]]
//       CHECK:   scf.for %[[IV3:.+]] = %[[BIDZ]] to %[[UB0]] step %[[NBLOCKSZ]]
//       CHECK:     scf.for %[[IV4:.+]] = %[[BOFFSETY]] to %[[UB3]] step %[[BSTEPY]]
//       CHECK:       scf.for %[[IV5:.+]] = %[[BOFFSETX]] to %[[UB4]] step %[[BSTEPX]]
//       CHECK:         %[[SV1:.+]] = subview %[[ARG1]][%[[IV3]], %[[IV4]], %[[IV5]], 0]
//       CHECK:         %[[SV2:.+]] = subview %[[ARG2]][%[[IV3]], %[[IV4]], %[[IV5]], 0]
//   CHECK-DAG:         %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//   CHECK-DAG:         %[[NTHREADSX:.+]] = "gpu.block_dim"() {dimension = "x"}
//   CHECK-DAG:         %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//   CHECK-DAG:         %[[NTHREADSY:.+]] = "gpu.block_dim"() {dimension = "y"}
//   CHECK-DAG:         %[[TIDZ:.+]] = "gpu.thread_id"() {dimension = "z"}
//   CHECK-DAG:         %[[NTHREADSZ:.+]] = "gpu.block_dim"() {dimension = "z"}
//       CHECK:         scf.for %{{.+}} = %[[TIDZ]] to %{{.+}} step %[[NTHREADSZ]]
//       CHECK:           scf.for %{{.+}} = %[[TIDY]] to %{{.+}} step %[[NTHREADSY]]
//       CHECK:             scf.for %{{.+}} = %[[TIDX]] to %{{.+}} step %[[NTHREADSX]]
//       CHECK:               scf.for
//       CHECK:                 scf.for
//       CHECK:                   scf.for
//       CHECK:                     scf.for
//   CHECK-NOT:                       linalg.conv

// -----

#map0 = affine_map<(d0)[s0, s1] -> (s0 + 4, -d0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> (s0 + 32, -d0 + s1)>
#map2 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map3 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map4 = affine_map<(d0)[s0] -> (32, -d0 + s0)>

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @pooling_no_padding(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}} {
    %c4 = constant 4 : index
    %c0 = constant 0 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index
    %0 = dim %arg1, %c0 : memref<?x?xf32>
    %1 = dim %arg1, %c1 : memref<?x?xf32>
    %2 = dim %arg2, %c0 : memref<?x?xf32>
    %3 = dim %arg2, %c1 : memref<?x?xf32>
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%2, %3) step (%c4, %c32) {
      %4 = dim %arg0, %c0 : memref<?x?xf32>
      %5 = affine.min #map0(%arg3)[%0, %4]
      %6 = dim %arg0, %c1 : memref<?x?xf32>
      %7 = affine.min #map1(%arg4)[%1, %6]
      %8 = subview %arg0[%arg3, %arg4] [%5, %7] [1, 1]  : memref<?x?xf32> to memref<?x?xf32, #map2>
      %9 = affine.min #map3(%arg3)[%2]
      %10 = affine.min #map4(%arg4)[%3]
      %11 = subview %arg2[%arg3, %arg4] [%9, %10] [1, 1]  : memref<?x?xf32> to memref<?x?xf32, #map2>
      linalg.pooling_max(%8, %arg1, %11) {__internal_linalg_transform__ = "workgroup", dilations = [1, 1], strides = [1, 1]} : memref<?x?xf32, #map2>, memref<?x?xf32>, memref<?x?xf32, #map2>
      scf.yield
    }
    return
  }
}

// CHECK-LABEL: func @pooling_no_padding
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//   CHECK-DAG:   %[[C32:.+]] = constant 32 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[UB0:.+]] = dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[UB1:.+]] = dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[UB2:.+]] = dim %[[ARG2]], %[[C0]]
//   CHECK-DAG:   %[[UB3:.+]] = dim %[[ARG2]], %[[C1]]
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[NBLOCKSX:.+]] = "gpu.grid_dim"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[NBLOCKSY:.+]] = "gpu.grid_dim"() {dimension = "y"}
//       CHECK:   %[[BOFFSETY:.+]] = muli %[[BIDY]], %[[C4]]
//       CHECK:   %[[BSTEPY:.+]] = muli %[[NBLOCKSY]], %[[C4]]
//       CHECK:   %[[BOFFSETX:.+]] = muli %[[BIDX]], %[[C32]]
//       CHECK:   %[[BSTEPX:.+]] = muli %[[NBLOCKSX]], %[[C32]]
//       CHECK:   scf.for %[[IV3:.+]] = %[[BOFFSETY]] to %[[UB2]] step %[[BSTEPY]]
//       CHECK:     scf.for %[[IV4:.+]] = %[[BOFFSETX]] to %[[UB3]] step %[[BSTEPX]]
//       CHECK:       %[[SV1:.+]] = subview %[[ARG0]][%[[IV3]], %[[IV4]]]
//       CHECK:       %[[SV2:.+]] = subview %[[ARG2]][%[[IV3]], %[[IV4]]]
//   CHECK-DAG:       %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//   CHECK-DAG:       %[[NTHREADSX:.+]] = "gpu.block_dim"() {dimension = "x"}
//   CHECK-DAG:       %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//   CHECK-DAG:       %[[NTHREADSY:.+]] = "gpu.block_dim"() {dimension = "y"}
//       CHECK:       scf.for %{{.+}} = %[[TIDY]] to %{{.+}} step %[[NTHREADSY]]
//       CHECK:         scf.for %{{.+}} = %[[TIDX]] to %{{.+}} step %[[NTHREADSX]]
//       CHECK:           scf.for
//       CHECK:             scf.for
//   CHECK-NOT:               linalg.pooling_max
