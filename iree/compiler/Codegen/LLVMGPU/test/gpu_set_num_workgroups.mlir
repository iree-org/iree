// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass{test-lowering-configuration}))' %s | IreeFileCheck %s

hal.executable @add_dispatch_0 {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @add_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
  builtin.module  {
    func @add_dispatch_0() {
      %c0 = constant 0 : index
      %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:16384xf32>
      %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:16384xf32>
      %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:16384xf32>
      %3 = linalg.init_tensor [16384] : tensor<16384xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16384xf32> -> tensor<16384xf32>
      %5 = flow.dispatch.tensor.load %1, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16384xf32> -> tensor<16384xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16384xf32>, tensor<16384xf32>) outs(%3 : tensor<16384xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16384xf32>
        flow.dispatch.tensor.store %6, %2, offsets=[], sizes=[], strides=[] : tensor<16384xf32> -> !flow.dispatch.tensor<writeonly:16384xf32>
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

//  CHECK-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[128], [], [4]{{\]}}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 128)>
//      CHECK: hal.executable.entry_point public @add_dispatch_0
// CHECK-SAME:     passPipeline = 3 : i32
// CHECK-SAME:     workloadPerWorkgroup = [128]
// CHECK-SAME:     workgroup_size = [32 : index, 1 : index, 1 : index]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index,
//  CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:     %[[NWGS_X:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//      CHECK:     hal.return %[[NWGS_X]], %[[C1]], %[[C1]]
//      CHECK: func @add_dispatch_0
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering.config = #[[CONFIG]]

// -----

hal.executable private @dot_dispatch_1  {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @dot_dispatch_1 attributes {interface = @legacy_io, ordinal = 0 : index}
    builtin.module  {
      func @dot_dispatch_1() {
        %c0 = constant 0 : index
        %c4 = constant 4 : index
        %c2 = constant 2 : index
        %cst = constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan @io::@ro0[%c0] : memref<2x3xf32>
        %1 = hal.interface.binding.subspan @io::@ro1[%c0] : memref<3x4xf32>
        %2 = hal.interface.binding.subspan @io::@wo2[%c0] : memref<2x4xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c2 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c4 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 2)>(%arg0)[%workgroup_size_y]
            %8 = memref.subview %0[%arg0, 0] [%7, 3] [1, 1] : memref<2x3xf32> to memref<?x3xf32, affine_map<(d0, d1)[s0] -> (d0 * 3 + s0 + d1)>>
            %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%arg1)[%workgroup_size_x]
            %10 = memref.subview %1[0, %arg1] [3, %9] [1, 1] : memref<3x4xf32> to memref<3x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>
            %11 = memref.subview %2[%arg0, %arg1] [%7, %9] [1, 1] : memref<2x4xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>
            linalg.fill(%cst, %11) : f32, memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>
            linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%8, %10 : memref<?x3xf32, affine_map<(d0, d1)[s0] -> (d0 * 3 + s0 + d1)>>, memref<3x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>) outs(%11 : memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + s0 + d1)>>)
          }
        }
        return
      }
      hal.interface private @legacy_io  {
        hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[4, 2, 4], [], [1, 1]{{\]}}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: hal.executable.entry_point public @dot_dispatch_1
// CHECK-SAME:     passPipeline = 4 : i32
// CHECK-SAME:     workloadPerWorkgroup = [2, 4]
// CHECK-SAME:     workgroup_size = [2 : index, 4 : index, 1 : index]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index,
//  CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:     %[[NWGS_X:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:     %[[NWGS_Y:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
//      CHECK:     hal.return %[[NWGS_X]], %[[NWGS_Y]], %[[C1]]
//      CHECK: func @dot_dispatch_1
//      CHECK:   linalg.fill
// CHECK-SAME:       lowering.config = #[[CONFIG]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering.config = #[[CONFIG]]

// -----

hal.executable @reduction_dispatch {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @predict_dispatch_153 attributes {
      interface = @io,
      ordinal = 0 : index}
    builtin.module  {
      func @predict_dispatch_153() {
        %c0 = constant 0 : index
        %cst = constant 0x7FC00000 : f32
        %cst_0 = constant 0xFF800000 : f32
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<1000xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : memref<f32>
        linalg.fill(%cst_0, %1) : f32, memref<f32>
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%0 : memref<1000xf32>) outs(%1 : memref<f32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %2 = cmpf ogt, %arg0, %arg1 : f32
          %3 = select %2, %arg0, %arg1 : f32
          %4 = cmpf uno, %arg0, %arg1 : f32
          %5 = select %4, %cst, %3 : f32
          linalg.yield %5 : f32
        }
        return
      }
      hal.interface private @io  {
        hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_xw_external, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG0:.+]] = {passPipeline = 2 : i32}
//  CHECK-DAG: #[[CONFIG1:.+]] = {tileSizes = {{\[}}[]{{\]}}}
//      CHECK: hal.executable.entry_point public @predict_dispatch_153
// CHECK-SAME:     translation.info = #[[CONFIG0]]
// CHECK-SAME:     workgroup_size = [1 : index, 1 : index, 1 : index]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index,
//  CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//      CHECK:     hal.return %[[C1]], %[[C1]], %[[C1]]
//      CHECK: linalg.fill
// CHECK-SAME:   lowering.config = #[[CONFIG1]]
//      CHECK: linalg.generic
// CHECK-SAME:   lowering.config = #[[CONFIG1]]

// -----

hal.executable @tensor_insert {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
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
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 128)>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:   translation.info = {passPipeline = 2 : i32, workloadPerWorkgroup = [128, 1]}
// CHECK-NEXT:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[NWGSX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//      CHECK:   hal.return %[[NWGSX]], %[[ARG1]], %[[C1]]

// -----

hal.executable @tensor_insert {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @tensor_insert_slice attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      builtin.func @tensor_insert_slice() {
        %c0 = constant 0 : index
        %d0 = hal.interface.load.constant offset = 0 : index
        %d1 = hal.interface.load.constant offset = 1 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<?x?xi32>{%d0, %d1}
        %1 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : memref<?x?xi32>{%d0, %d1}
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %2 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_id_y]
        %3 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_count_y]
        scf.for %arg0 = %2 to %d0 step %3 {
          %4 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %d0]
          %5 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_id_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_count_x]
          scf.for %arg1 = %5 to %d1 step %6 {
            %7 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %d1]
            %8 = memref.subview %0[%arg0, %arg1] [%4, %7] [1, 1] : memref<?x?xi32> to memref<?x?xi32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            %9 = affine.apply affine_map<(d0) -> (d0 + 4)>(%arg0)
            %10 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg1)
            %11 = memref.subview %1[%9, %10] [%4, %7] [1, 1] : memref<?x?xi32> to memref<?x?xi32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>
            linalg.copy(%8, %11) : memref<?x?xi32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>, memref<?x?xi32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>> 
          }
        }
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[1, 128], [], [1, 4]{{\]}}}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 128)>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:   translation.info = {passPipeline = 3 : i32, workloadPerWorkgroup = [128, 1]}
// CHECK-NEXT:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[NWGSX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//      CHECK:   hal.return %[[NWGSX]], %[[ARG1]], %[[C1]]
//      CHECK:   linalg.copy
// CHECK-SAME:     lowering.config = #[[CONFIG]]
