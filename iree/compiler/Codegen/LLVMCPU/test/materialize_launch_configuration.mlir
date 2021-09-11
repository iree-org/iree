// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true}))' -cse -canonicalize -split-input-file %s | IreeFileCheck %s

hal.executable private @matmul_tensors  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {
       data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
       native_vector_size = 16 : index,
       target_triple = "x86_64-unknown-linux-gnu"
    }> {
    hal.executable.entry_point @matmul_tensors attributes {
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module {
      func @matmul_tensors() {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %pcM = hal.interface.load.constant offset = 0 : index
        %pcN = hal.interface.load.constant offset = 1 : index
        %pcK = hal.interface.load.constant offset = 2 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?xf32>{%pcM, %pcK}
        %2 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?x?xf32>{%pcK, %pcN}
        %4 = hal.interface.binding.subspan @io::@arg2[%c0] : memref<?x?xf32>{%pcM, %pcN}
        %6 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?xf32>{%pcM, %pcN}
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

//  CHECK-DAG: #[[CONFIG:.+]] = {nativeVectorSize = [4, 4, 4], tileSizes = {{\[}}[64, 64], [32, 32, 32], [4, 4, 4]{{\]}}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//      CHECK: hal.executable.entry_point public @matmul_tensors
// CHECK-NEXT:   (%[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:    hal.return %[[D0]], %[[D1]], %[[C1]] : index, index, index
//      CHECK: linalg.matmul
// CHECK-SAME:   lowering.config = #[[CONFIG]]

// -----

//      CHECK: #[[CONFIG:.+]] = {passPipeline = 0 : i32}
//  CHECK-NOT: #config
//      CHECK: hal.executable.entry_point public @add_no_config
// CHECK-SAME:     translation.info = #[[CONFIG]]
//  CHECK-NOT:     #config

hal.executable private @add_no_config  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {
       data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
       native_vector_size = 16 : index,
       target_triple = "x86_64-unknown-linux-gnu"
    }> {
    hal.executable.entry_point @add_no_config attributes {
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module  {
      func @add_no_config() {
        %c0 = constant 0 : index
        %dim0 = hal.interface.load.constant offset = 0 : index
        %dim1 = hal.interface.load.constant offset = 1 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?xf32>{%dim0, %dim1}
        %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?xf32>{%dim1}
        %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?xf32>{%dim0, %dim1}
        linalg.generic {
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
      hal.interface private @io  {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// -----

hal.executable private @add  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {
       data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
       native_vector_size = 16 : index,
       target_triple = "x86_64-unknown-linux-gnu"
    }> {
    hal.executable.entry_point @add attributes {
      interface = @io, ordinal = 0 : index,
      signature = (!flow.dispatch.tensor<readonly:?x?xf32>, !flow.dispatch.tensor<readonly:?xf32>,
        !flow.dispatch.tensor<writeonly:?x?xf32>) -> ()}
    builtin.module  {
      func @add() {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %dim0 = hal.interface.load.constant offset = 0 : index
        %dim1 = hal.interface.load.constant offset = 1 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?xf32>{%dim0, %dim1}
        %2 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?xf32>{%dim1}
        %6 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?xf32>{%dim0, %dim1}
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
      hal.interface private @io  {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[64, 64]{{\]}}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//      CHECK: hal.executable.entry_point public @add
// CHECK-NEXT:   (%[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:    hal.return %[[D0]], %[[D1]], %[[C1]] : index, index, index
//      CHECK: linalg.generic
// CHECK-SAME:  lowering.config = #[[CONFIG]]

// -----

hal.executable private @add4D  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {
       data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
       native_vector_size = 16 : index,
       target_triple = "x86_64-unknown-linux-gnu"
    }> {
    hal.executable.entry_point @add4D attributes {
      interface = @io, ordinal = 0 : index,
      signature = (!flow.dispatch.tensor<readonly:?x?x?x?xf32>, !flow.dispatch.tensor<readonly:?xf32>,
        !flow.dispatch.tensor<writeonly:?x?x?x?xf32>) -> ()}
    builtin.module  {
      func @add4D() {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %c2 = constant 2 : index
        %c3 = constant 3 : index
        %dim0 = hal.interface.load.constant offset = 0 : index
        %dim1 = hal.interface.load.constant offset = 1 : index
        %dim2 = hal.interface.load.constant offset = 2 : index
        %dim3 = hal.interface.load.constant offset = 3 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?x?x?xf32>{%dim0, %dim1, %dim2, %dim3}
        %2 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?x?x?x?xf32>{%dim0, %dim1, %dim2, %dim3}
        %6 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?x?x?xf32>{%dim0, %dim1, %dim2, %dim3}
        %B = memref.dim %0, %c0 : memref<?x?x?x?xf32>
        %M = memref.dim %0, %c1 : memref<?x?x?x?xf32>
        %N = memref.dim %0, %c2 : memref<?x?x?x?xf32>
        %K = memref.dim %0, %c3 : memref<?x?x?x?xf32>
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
        scf.for %arg20 = %28 to %M step %29 {
          %8 = muli %workgroup_size_y, %workgroup_id_y : index
          %9 = muli %workgroup_size_y, %workgroup_count_y : index
          scf.for %arg0 = %8 to %N step %9 {
            %10 = muli %workgroup_size_x, %workgroup_id_x : index
            %11 = muli %workgroup_size_x, %workgroup_count_x : index
            scf.for %arg1 = %10 to %K step %11 {
              %212 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg20)[%workgroup_size_z, %M]
              %12 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %N]
              %13 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %K]
              %14 = memref.subview %0[0, %arg20, %arg0, %arg1] [%B, %212, %12, %13] [1, 1, 1, 1]
                : memref<?x?x?x?xf32> to
                  memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>>
              %15 = memref.subview %2[0, %arg20, %arg0, %arg1] [%B, %212, %12, %13] [1, 1, 1, 1]
                : memref<?x?x?x?xf32> to
                  memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>>
              %16 = memref.subview %6[0, %arg20, %arg0, %arg1] [%B, %212, %12, %13] [1, 1, 1, 1]
                 : memref<?x?x?x?xf32> to
                   memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>>
              linalg.generic {
                __internal_linalg_transform__ = "workgroup",
                indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                 affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                 affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
                ins(%14, %15 :
                      memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>>,
                      memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>>)
                outs(%16 :  memref<?x?x?x?xf32, affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>>) {
                ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
                  %3 = addf %arg2, %arg3 : f32
                  linalg.yield %3 : f32
                }
              }
            }
          }
          return
        }
        hal.interface private @io  {
          hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
          hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
          hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
        }
      }
    }
  }
//  CHECK-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[0, 64, 64, 64]{{\]}}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//      CHECK: hal.executable.entry_point public @add4D
// CHECK-NEXT:   (%[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//  CHECK-DAG:    %[[D2:.+]] = affine.apply #[[MAP0]]()[%[[ARG2]]]
//      CHECK:    hal.return %[[D0]], %[[D1]], %[[D2]] : index, index, index
//      CHECK: linalg.generic
// CHECK-SAME:  lowering.config = #[[CONFIG]]

// -----

hal.executable private @batch_matmul_tensors  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {
       data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
       native_vector_size = 16 : index,
       target_triple = "x86_64-unknown-linux-gnu"
    }> {
    hal.executable.entry_point @batch_matmul_tensors attributes {
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module {
      func @batch_matmul_tensors() {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %c2 = constant 2 : index
        %pcB = hal.interface.load.constant offset = 0 : index
        %pcM = hal.interface.load.constant offset = 1 : index
        %pcN = hal.interface.load.constant offset = 2 : index
        %pcK = hal.interface.load.constant offset = 3 : index
        %0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?x?xf32>{%pcB, %pcM, %pcK}
        %2 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?x?x?xf32>{%pcB, %pcK, %pcN}
        %4 = hal.interface.binding.subspan @io::@arg2[%c0] : memref<?x?x?xf32>{%pcB, %pcM, %pcN}
        %6 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?x?xf32>{%pcB, %pcM, %pcN}
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
//  CHECK-DAG: #[[CONFIG:.+]] = {nativeVectorSize = [1, 4, 4, 4], tileSizes = {{\[}}[1, 32, 32], [1, 16, 16, 16], [1, 4, 4, 4]{{\]}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//      CHECK: hal.executable.entry_point public @batch_matmul_tensors
// CHECK-NEXT: (%[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: index)
//  CHECK-DAG:  %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:  %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:  hal.return %[[D0]], %[[D1]], %[[ARG2]]
//      CHECK:  linalg.batch_matmul
// CHECK-SAME:    lowering.config = #[[CONFIG]]

// -----

hal.executable private @preset_config_matmul_tensors  {
  hal.executable.variant @system_elf_x86_64, target = #hal.executable.target<"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @preset_config attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      builtin.func @preset_config() {
        %c0 = constant 0 : index
        %c512 = constant 512 : index
        %c128 = constant 128 : index
        %cst = constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:128x256xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:256x512xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:128x512xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c128 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c512 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 128)>(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<?x256xf32>
            %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 512)>(%arg1)[%workgroup_size_x]
            %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [256, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x512xf32> -> tensor<256x?xf32>
            %11 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 128)>(%arg0)[%workgroup_size_y]
            %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 512)>(%arg1)[%workgroup_size_x]
            %13 = affine.min affine_map<(d0)[s0] -> (-d0 + 128, s0)>(%arg0)[%workgroup_size_y]
            %14 = affine.min affine_map<(d0)[s0] -> (-d0 + 512, s0)>(%arg1)[%workgroup_size_x]
            %15 = linalg.init_tensor [%13, %14] : tensor<?x?xf32>
            %16 = linalg.fill(%cst, %15) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
            %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup", lowering.config = {passPipeline = 1 : i32, tileSizes = [[32, 32, 32]]}} ins(%8, %10 : tensor<?x256xf32>, tensor<256x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
            flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:128x512xf32>
          }
        }
        return
      }
      hal.interface private @io  {
        hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = {nativeVectorSize = [], tileSizes = {{\[}}[32, 32, 32]{{\]}}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//      CHECK: hal.executable.entry_point
// CHECK-SAME:     translation.info = {passPipeline = 1 : i32, workloadPerWorkgroup = [32, 32]}
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:     %[[NWG_X:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:     %[[NWG_Y:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:     return %[[NWG_X]], %[[NWG_Y]], %[[C1]]
//      CHECK: builtin.module
//      CHECK:   func @preset_config
//  CHECK-DAG:     %[[WGID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:     %[[WGCOUNT_X:.+]] = hal.interface.workgroup.count[0]
//  CHECK-DAG:     %[[WGID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:     %[[WGCOUNT_Y:.+]] = hal.interface.workgroup.count[1]
//      CHECK:     %[[LB_Y:.+]] = affine.apply #[[MAP1]]()[%[[WGID_Y]]]
//      CHECK:     %[[STEP_Y:.+]] = affine.apply #[[MAP1]]()[%[[WGCOUNT_Y]]]
//      CHECK:     scf.for %[[IV0:.+]] = %[[LB_Y]] to %{{.+}} step %[[STEP_Y]]
//      CHECK:       %[[LB_X:.+]] = affine.apply #[[MAP1]]()[%[[WGID_X]]]
//      CHECK:       %[[STEP_X:.+]] = affine.apply #[[MAP1]]()[%[[WGCOUNT_X]]]
//      CHECK:       scf.for %[[IV1:.+]] = %[[LB_X]] to %{{.+}} step %[[STEP_X]]
//      CHECK:         linalg.matmul
// CHECK-SAME:             lowering.config = #[[CONFIG]]
// CHECK-SAME:             ins(%{{.+}}, %{{.+}} : tensor<32x256xf32>, tensor<256x32xf32>)
// CHECK-SAME:             outs(%{{.+}} : tensor<32x32xf32>)

// -----

hal.executable @tensor_insert {
  hal.executable.variant @system_elf_x86_64, target = #hal.executable.target<"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @tensor_insert_slice attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      builtin.func @tensor_insert_slice() {
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:?x?xi32>
        %1 = hal.interface.load.constant offset = 0 : index
        %2 = hal.interface.load.constant offset = 1 : index
        %3 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : !flow.dispatch.tensor<writeonly:?x?xi32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_id_y]
        %5 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_count_y]
        %d0 = hal.interface.load.constant offset = 2 : index
        %d1 = hal.interface.load.constant offset = 2 : index
        scf.for %arg0 = %4 to %d0 step %5 {
          %6 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %d0]
          %7 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_id_x]
          %8 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_count_x]
          scf.for %arg1 = %7 to %d1 step %8 {
            %9 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %d1]
            %10 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [%6, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xi32> -> tensor<?x?xi32>
            %11 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg0)[%1]
            %12 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg1)[%2]
            flow.dispatch.tensor.store %10, %3, offsets = [%11, %12], sizes = [%6, %9], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?xi32>
          }
        }
        return
      }
      hal.interface @io attributes {push_constants = 2 : index, sym_visibility = "private"} {
        hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_xw_external, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:   translation.info = {passPipeline = 0 : i32, workloadPerWorkgroup = [64, 64]}
// CHECK-NEXT:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[NWGSX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:   %[[NWGSY:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:   hal.return %[[NWGSX]], %[[NWGSY]], %[[C1]]
