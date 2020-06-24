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
//   CHECK-DAG:     %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:     %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:     %[[C3:.+]] = constant 3 : index
//   CHECK-DAG:     %[[UB0:.+]] = dim %{{.*}}, %[[C0]]
//   CHECK-DAG:     %[[UB1:.+]] = dim %{{.*}}, %[[C1]]
//   CHECK-DAG:     %[[UB2:.+]] = dim %{{.*}}, %[[C2]]
//   CHECK-DAG:     %[[UB3:.+]] = dim %{{.*}}, %[[C3]]
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
//       CHECK:       load %{{.*}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:       load %{{.*}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:       store %{{.*}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]


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
//   CHECK-DAG:     %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:     %[[C2:.+]] = constant 2 : index
//       CHECK:     %[[UB0:.+]] = dim %{{.*}}, %[[C0]]
//       CHECK:     %[[UB1:.+]] = dim %{{.*}}, %[[C1]]
//       CHECK:     %[[UB2:.+]] = dim %{{.*}}, %[[C2]]
//       CHECK:     %[[UB:.+]] = muli %[[UB1]], %[[UB0]]
//       CHECK:     %[[COND:.+]] = cmpi "slt", %{{.*}}, %[[UB]]
//       CHECK:     scf.if %[[COND]]
//       CHECK:       %[[IV0:.+]] = divi_signed %{{.*}}, %[[UB1]]
//       CHECK:       %[[IV1:.+]] = remi_signed %{{.*}}, %[[UB1]]
//       CHECK:       scf.for %[[IV:.+]] = %{{.*}} to %[[UB2]]
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
        linalg.matmul %5, %9, %14 {__internal_linalg_transform__ = "workitem"} : (memref<?x?xf32, #map2>, memref<?x?xf32, #map2>, memref<?x?xf32, #map2>)
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
//   CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//   CHECK-DAG:   %[[C8:.+]] = constant 8 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[UB0:.+]] = dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[UB1:.+]] = dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[UB2:.+]] = dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[BIDX:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:   %[[GDIMX:.+]] = "gpu.grid_dim"() {dimension = "x"}
//   CHECK-DAG:   %[[BIDY:.+]] = "gpu.block_id"() {dimension = "y"}
//   CHECK-DAG:   %[[GDIMY:.+]] = "gpu.grid_dim"() {dimension = "y"}
//       CHECK:   %[[BOFFSETY:.+]] = muli %[[BIDY]], %[[C8]]
//       CHECK:   %[[BSTEPY:.+]] = muli %[[GDIMY]], %[[C8]]
//       CHECK:   %[[BOFFSETX:.+]] = muli %[[BIDX]], %[[C8]]
//       CHECK:   %[[BSTEPX:.+]] = muli %[[GDIMX]], %[[C8]]
//       CHECK:   scf.for %[[BIV0:.+]] = %[[BOFFSETY]] to %[[UB0]] step %[[BSTEPY]]
//       CHECK:     scf.for %[[BIV1:.+]] = %[[BOFFSETX]] to %[[UB1]] step %[[BSTEPX]]
//       CHECK:       scf.for %[[BIV2:.+]] = %[[C0]] to %[[UB2]] step %[[C4]]
//   CHECK-DAG:         %[[VIEWUB0:.+]] = affine.min #{{.*}}(%[[BIV0]])[%[[UB0]]]
//   CHECK-DAG:         %[[VIEWUB1:.+]] = affine.min #{{.*}}(%[[BIV1]])[%[[UB1]]]
//   CHECK-DAG:         %[[VIEWUB2:.+]] = affine.min #{{.*}}(%[[BIV2]])[%[[UB2]]]
//   CHECK-DAG:         %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"}
//   CHECK-DAG:         %[[TIDY:.+]] = "gpu.thread_id"() {dimension = "y"}
//       CHECK:         %[[INBOUNDY:.+]] = cmpi "slt", %[[TIDY]], %[[VIEWUB0]]
//       CHECK:         %[[INBOUNDX:.+]] = cmpi "slt", %[[TIDX]], %[[VIEWUB1]]
//       CHECK:         %[[COND:.+]] = and %[[INBOUNDY]], %[[INBOUNDX]]
//       CHECK:         scf.if %[[COND]]
//       CHECK:           scf.for %{{.*}} = %[[C0]] to %[[VIEWUB2]] step %[[C1]]
