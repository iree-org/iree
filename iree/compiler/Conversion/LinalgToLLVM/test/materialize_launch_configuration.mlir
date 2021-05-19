// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-llvm-materialize-launch-configuration))" -cse -canonicalize -split-input-file %s | IreeFileCheck %s
// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-llvm-materialize-launch-configuration))" -cse -canonicalize -split-input-file -iree-llvm-tile-size=4,2,1 %s | IreeFileCheck --check-prefix=CHECK_FLAG %s

hal.executable @matmul_tensors attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @matmul_tensors attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module {
      func @matmul_tensors() {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?xf32>
        %2 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?x?xf32>
        %4 = hal.interface.binding.subspan @io::@arg2[%c0] : memref<?x?xf32>
        %6 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?xf32>
        %M = memref.dim %0, %c0 : memref<?x?xf32>
        %N = memref.dim %2, %c1 : memref<?x?xf32>
        %K = memref.dim %0, %c1 : memref<?x?xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %8 = muli %workgroup_size_y, %workgroup_id_y : index
        %9 = muli %workgroup_size_y, %workgroup_count_y : index
        scf.for %arg0 = %8 to %M step %9 {
          %10 = muli %workgroup_size_x, %workgroup_id_x : index
          %11 = muli %workgroup_size_x, %workgroup_count_x : index
          scf.for %arg1 = %10 to %N step %11 {
            %12 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %N]
            %13 = memref.subview %0[%arg0, 0] [%12, %K] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %14 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %M]
            %15 = memref.subview %2[0, %arg1] [%K, %14] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %16 = memref.subview %4[%arg0, %arg1] [%12, %14] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %17 = memref.alloc(%12, %14) : memref<?x?xf32>
            linalg.copy(%16, %17) : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>, memref<?x?xf32>
            linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%13, %15 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>, memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>) outs(%17 : memref<?x?xf32>)
            %18 = memref.subview %6[%arg0, %arg1] [%12, %14] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            linalg.copy(%17, %18) : memref<?x?xf32>, memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG0:.+]] = {tileSizes = {{\[}}[64, 64]{{\]}}}
//  CHECK-DAG: #[[CONFIG1:.+]] = {nativeVectorSize = [4, 4, 4], tileSizes = {{\[}}[64, 64], [32, 32, 32], [4, 4, 4]{{\]}}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//      CHECK: hal.executable.entry_point @matmul_tensors
// CHECK-NEXT:   (%[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:    hal.return %[[D0]], %[[D1]], %[[C1]] : index, index, index
//      CHECK: linalg.copy
// CHECK-SAME:   lowering.config = #[[CONFIG0]]
//      CHECK: linalg.matmul
// CHECK-SAME:   lowering.config = #[[CONFIG1]]
//      CHECK: linalg.copy
// CHECK-SAME:   lowering.config = #[[CONFIG0]]

//  CHECK_FLAG-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[4, 2]{{\]}}}
//  CHECK_FLAG-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//  CHECK_FLAG-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK_FLAG: hal.executable.entry_point @matmul_tensors
// CHECK_FLAG-NEXT:   (%[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK_FLAG-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK_FLAG-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: index)
//  CHECK_FLAG-DAG:    %[[C1:.+]] = constant 1 : index
//  CHECK_FLAG-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK_FLAG-DAG:    %[[D1:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
//      CHECK_FLAG:    hal.return %[[D0]], %[[D1]], %[[C1]] : index, index, index
//      CHECK_FLAG: linalg.copy
// CHECK_FLAG-SAME:   lowering.config = #[[CONFIG]]
//      CHECK_FLAG: linalg.matmul
// CHECK_FLAG-SAME:   lowering.config = #[[CONFIG]]
//      CHECK_FLAG: linalg.copy
// CHECK_FLAG-SAME:   lowering.config = #[[CONFIG]]

// -----

// CHECK-NOT: #config
// CHECK_FLAG-NOT: #config

hal.executable @add_no_config attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @add_no_config attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module  {
      func @add_no_config() {
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?xf32>
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?xf32>
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?xf32>
        linalg.generic {__internal_linalg_transform__ = "workgroup"} {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%0, %1 : memref<?x?xf32>, memref<?xf32>) outs(%2 : memref<?x?xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
            %3 = addf %arg0, %arg1 : f32
            linalg.yield %3 : f32
          }
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// -----

hal.executable @add attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @add attributes {
      interface = @io, ordinal = 0 : index,
      signature = (!flow.dispatch.tensor<readonly:?x?xf32>, !flow.dispatch.tensor<readonly:?xf32>,
        !flow.dispatch.tensor<writeonly:?x?xf32>) -> ()}
    module  {
      func @add() {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?xf32>
        %2 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?xf32>
        %6 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?xf32>
        %M = memref.dim %0, %c0 : memref<?x?xf32>
        %N = memref.dim %0, %c1 : memref<?x?xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %8 = muli %workgroup_size_y, %workgroup_id_y : index
        %9 = muli %workgroup_size_y, %workgroup_count_y : index
        scf.for %arg0 = %8 to %M step %9 {
          %10 = muli %workgroup_size_x, %workgroup_id_x : index
          %11 = muli %workgroup_size_x, %workgroup_count_x : index
          scf.for %arg1 = %10 to %N step %11 {
            %12 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %M]
            %13 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %N]
            %14 = memref.subview %0[%arg0, %arg1] [%12, %13] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %15 = memref.subview %2[%arg1] [%13] [1] : memref<?xf32> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
            %16 = memref.subview %6[%arg0, %arg1] [%12, %13] [1, 1] : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            linalg.generic {
              __internal_linalg_transform__ = "workgroup",
              indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                               affine_map<(d0, d1) -> (d1)>,
                               affine_map<(d0, d1) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel"]}
              ins(%14, %15 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>, memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>) outs(%16 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>) {
              ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
                %3 = addf %arg2, %arg3 : f32
                linalg.yield %3 : f32
              }
          }
        }
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[128, 128]{{\]}}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 128)>
//      CHECK: hal.executable.entry_point @add
// CHECK-NEXT:   (%[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:    hal.return %[[D0]], %[[D1]], %[[C1]] : index, index, index
//      CHECK: linalg.generic
// CHECK-SAME:  lowering.config = #[[CONFIG]]

//  CHECK_FLAG-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[4, 2]{{\]}}}
//  CHECK_FLAG-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//  CHECK_FLAG-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK_FLAG: hal.executable.entry_point @add
// CHECK_FLAG-NEXT:   (%[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK_FLAG-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK_FLAG-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: index)
//  CHECK_FLAG-DAG:    %[[C1:.+]] = constant 1 : index
//  CHECK_FLAG-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK_FLAG-DAG:    %[[D1:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
//      CHECK_FLAG:    hal.return %[[D0]], %[[D1]], %[[C1]] : index, index, index
//      CHECK_FLAG: linalg.generic
// CHECK_FLAG-SAME:  lowering.config = #[[CONFIG]]

// -----

hal.executable @batch_matmul_tensors attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @batch_matmul_tensors attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module {
      func @batch_matmul_tensors() {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %c2 = constant 2 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?x?xf32>
        %2 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?x?x?xf32>
        %4 = hal.interface.binding.subspan @io::@arg2[%c0] : memref<?x?x?xf32>
        %6 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?x?xf32>
        %M = memref.dim %0, %c1 : memref<?x?x?xf32>
        %N = memref.dim %2, %c2 : memref<?x?x?xf32>
        %B = memref.dim %0, %c0 : memref<?x?x?xf32>
        %K = memref.dim %0, %c2 : memref<?x?x?xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %28 = muli %workgroup_size_z, %workgroup_id_z : index
        %29 = muli %workgroup_size_z, %workgroup_count_z : index
        scf.for %arg20 = %28 to %B step %29 {
          %8 = muli %workgroup_size_y, %workgroup_id_y : index
          %9 = muli %workgroup_size_y, %workgroup_count_y : index
          scf.for %arg0 = %8 to %M step %9 {
            %10 = muli %workgroup_size_x, %workgroup_id_x : index
            %11 = muli %workgroup_size_x, %workgroup_count_x : index
            scf.for %arg1 = %10 to %N step %11 {
              %212 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg20)[%workgroup_size_z, %B]
              %12 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %N]
              %13 = memref.subview %0[%arg20, %arg0, 0] [%212, %12, %K] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>>
              %14 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %M]
              %15 = memref.subview %2[%arg20, 0, %arg1] [%212, %K, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>>
              %16 = memref.subview %4[%arg20, %arg0, %arg1] [%212, %12, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>>
              %17 = memref.alloc(%212, %12, %14) : memref<?x?x?xf32>
              linalg.copy(%16, %17) : memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>>, memref<?x?x?xf32>
              linalg.batch_matmul {__internal_linalg_transform__ = "workgroup"} ins(%13, %15 : memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>>, memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>>) outs(%17 : memref<?x?x?xf32>)
              %18 = memref.subview %6[%arg20, %arg0, %arg1] [%212, %12, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>>
              linalg.copy(%17, %18) : memref<?x?x?xf32>, memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>>
            }
          }
        }
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG0:.+]] = {tileSizes = {{\[}}[1, 32, 32]{{\]}}
//  CHECK-DAG: #[[CONFIG1:.+]] = {nativeVectorSize = [1, 4, 4, 4], tileSizes = {{\[}}[1, 32, 32], [1, 16, 16, 16], [1, 4, 4, 4]{{\]}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//      CHECK: hal.executable.entry_point @batch_matmul_tensors
// CHECK-NEXT: (%[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: index)
//  CHECK-DAG:  %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:  %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:  hal.return %[[D0]], %[[D1]], %[[ARG2]]
//      CHECK:  linalg.copy
// CHECK-SAME:    lowering.config = #[[CONFIG0]]
//      CHECK:  linalg.batch_matmul
// CHECK-SAME:    lowering.config = #[[CONFIG1]]
//      CHECK:  linalg.copy
// CHECK-SAME:    lowering.config = #[[CONFIG0]]

//  CHECK_FLAG-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[4, 2, 1]{{\]}}
//  CHECK_FLAG-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//  CHECK_FLAG-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK_FLAG: hal.executable.entry_point @batch_matmul_tensors
// CHECK_FLAG-NEXT: (%[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK_FLAG-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK_FLAG-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: index)
//  CHECK_FLAG-DAG:  %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//  CHECK_FLAG-DAG:  %[[D1:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]]]
//      CHECK_FLAG:  hal.return %[[ARG0]], %[[D0]], %[[D1]]
//      CHECK_FLAG:  linalg.copy
// CHECK_FLAG-SAME:    lowering.config = #[[CONFIG]]
//      CHECK_FLAG:  linalg.batch_matmul
// CHECK_FLAG-SAME:    lowering.config = #[[CONFIG]]
//      CHECK_FLAG:  linalg.copy
// CHECK_FLAG-SAME:    lowering.config = #[[CONFIG]]
