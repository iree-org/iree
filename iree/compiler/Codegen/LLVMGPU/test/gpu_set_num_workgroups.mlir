// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass{test-lowering-configuration}))' %s | IreeFileCheck %s

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
hal.executable @add_dispatch_0 attributes {sym_visibility = "private"} {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @add_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
    module  {
      func @add_dispatch_0() {
        %c0 = constant 0 : index
        %c1024 = constant 1024 : index
        %0 = hal.interface.binding.subspan @io::@ro0[%c0] : memref<1024xf32>
        %1 = hal.interface.binding.subspan @io::@ro1[%c0] : memref<1024xf32>
        %2 = hal.interface.binding.subspan @io::@wo2[%c0] : memref<1024xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %3 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
        %4 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
        scf.for %arg0 = %3 to %c1024 step %4 {
          %5 = affine.min #map3(%arg0)[%workgroup_size_x]
          %6 = memref.subview %0[%arg0] [%5] [1] : memref<1024xf32> to memref<?xf32, #map1>
          %7 = memref.subview %1[%arg0] [%5] [1] : memref<1024xf32> to memref<?xf32, #map1>
          %8 = memref.subview %2[%arg0] [%5] [1] : memref<1024xf32> to memref<?xf32, #map1>
          linalg.generic {
              indexing_maps = [#map2, #map2, #map2],
              iterator_types = ["parallel"]}
              ins(%6, %7 : memref<?xf32, #map1>, memref<?xf32, #map1>)
              outs(%8 : memref<?xf32, #map1>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
            ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
              %9 = addf %arg1, %arg2 : f32
              linalg.yield %9 : f32
          }
        }
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[128], [], [4]{{\]}}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 128)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 128)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0) -> (d0)>
//      CHECK: hal.executable.entry_point @add_dispatch_0
// CHECK-SAME:     passPipeline = 2 : i32
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

hal.executable @dot_dispatch_1 attributes {sym_visibility = "private"} {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @dot_dispatch_1 attributes {interface = @legacy_io, ordinal = 0 : index}
    module  {
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
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[2, 256, 4], [], [2, 4]{{\]}}}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 256)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//      CHECK: hal.executable.entry_point @dot_dispatch_1
// CHECK-SAME:     passPipeline = 4 : i32
// CHECK-SAME:     workloadPerWorkgroup = [256, 2]
// CHECK-SAME:     workgroup_size = [64 : index, 1 : index, 1 : index]
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
    module  {
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
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_xw_external, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = {tileSizes = {{\[}}[]{{\]}}}
//      CHECK: hal.executable.entry_point @predict_dispatch_153
// CHECK-SAME:     passPipeline = 2 : i32
// CHECK-SAME:     workgroup_size = [1 : index, 1 : index, 1 : index]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index,
//  CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//      CHECK:     hal.return %[[C1]], %[[C1]], %[[C1]]
//      CHECK: linalg.fill
// CHECK-SAME:   lowering.config = #[[CONFIG]]
//      CHECK: linalg.generic
// CHECK-SAME:   lowering.config = #[[CONFIG]]
