// RUN: iree-opt -split-input-file  -pass-pipeline="hal.executable(hal.executable.variant(iree-llvmgpu-tile-and-distribute))" %s | IreeFileCheck %s

hal.executable @add_dispatch_0 attributes {sym_visibility = "private"} {
hal.executable.variant @cuda, target="cuda" {
  hal.executable.entry_point @add_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
  module  {
    func @add_dispatch_0() attributes {llvmgpu_workgroup_size = dense<[32, 1, 1]> : vector<3xi64>} {
      %c0 = constant 0 : index
      %c1024 = constant 1024 : index
      %0 = hal.interface.binding.subspan @io::@ro0[%c0] : memref<1024xf32>
      %1 = hal.interface.binding.subspan @io::@ro1[%c0] : memref<1024xf32>
      %2 = hal.interface.binding.subspan @io::@wo2[%c0] : memref<1024xf32>
      %workgroup_size_x = hal.interface.workgroup.size[0] : index
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
      %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
      scf.for %arg0 = %3 to %c1024 step %4 {
        %5 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg0)[%workgroup_size_x]
        %6 = memref.subview %0[%arg0] [%5] [1] : memref<1024xf32> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
        %7 = memref.subview %1[%arg0] [%5] [1] : memref<1024xf32> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
        %8 = memref.subview %2[%arg0] [%5] [1] : memref<1024xf32> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
        linalg.generic {lowering.config = {tileSizes = [[128], [], [4]]}, indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%6, %7 : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>, memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>) outs(%8 : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
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

// CHECK-LABEL: func @add_dispatch_0()
//  CHECK-SAME: llvmgpu_workgroup_size = dense<[32, 1, 1]>
//      CHECK:    "gpu.thread_id"() {dimension = "x"}
//      CHECK:    scf.for
//      CHECK:      linalg.generic

// -----

hal.executable @dot_dispatch_0 attributes {sym_visibility = "private"} {
hal.executable.variant @cuda, target="cuda" {
  hal.executable.entry_point @dot_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : index}
  module  {
    func @dot_dispatch_0() attributes {llvmgpu_workgroup_size = dense<[64, 1, 1]> : vector<3xi64>} {
      %cst = constant 0.000000e+00 : f32
      %c0 = constant 0 : index
      %c1024 = constant 1024 : index
      %0 = hal.interface.binding.subspan @legacy_io::@ro0[%c0] : memref<1024x1024xf32>
      %1 = hal.interface.binding.subspan @legacy_io::@ro1[%c0] : memref<1024x1024xf32>
      %2 = hal.interface.binding.subspan @legacy_io::@wo2[%c0] : memref<1024x1024xf32>
      %workgroup_size_x = hal.interface.workgroup.size[0] : index
      %workgroup_size_y = hal.interface.workgroup.size[1] : index
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %workgroup_count_y = hal.interface.workgroup.count[1] : index
      %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
      %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
      scf.for %arg0 = %3 to %c1024 step %4 {
        %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
        %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
        scf.for %arg1 = %5 to %c1024 step %6 {
          %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg0)[%workgroup_size_y]
          %8 = memref.subview %0[%arg0, 0] [%7, 1024] [1, 1] : memref<1024x1024xf32> to memref<?x1024xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
          %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg1)[%workgroup_size_x]
          %10 = memref.subview %1[0, %arg1] [1024, %9] [1, 1] : memref<1024x1024xf32> to memref<1024x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
          %11 = memref.subview %2[%arg0, %arg1] [%7, %9] [1, 1] : memref<1024x1024xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
          linalg.fill(%cst, %11) {__internal_linalg_transform__ = "workgroup", lowering.config = {tileSizes = [[2, 256, 4], [], [2, 4]]}} : f32, memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
          linalg.matmul {__internal_linalg_transform__ = "workgroup", lowering.config = {tileSizes = [[2, 256, 4], [], [2, 4]]}} ins(%8, %10 : memref<?x1024xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>, memref<1024x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>) outs(%11 : memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
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

//   CHECK-LABEL: hal.executable @dot_dispatch_0
//         CHECK:   hal.executable.variant @cuda, target="cuda" {
//         CHECK:  memref.global "private" @{{.*}} : memref<4x256xf32, 3>
//         CHECK:  memref.global "private" @{{.*}} : memref<2x4xf32, 3>
//     CHECK-DAG:  %[[C1024:.+]] = constant 1024 : index
//     CHECK-DAG:  %[[C0:.+]] = constant 0 : index
//     CHECK-DAG:  %[[C1:.+]] = constant 1 : index
//     CHECK-DAG:  %[[C2:.+]] = constant 2 : index
//     CHECK-DAG:  %[[C4:.+]] = constant 4 : index
//     CHECK-DAG:  %[[C256:.+]] = constant 256 : index
//         CHECK:  scf.for %[[K:.+]] = %[[C0]] to %[[C1024]] step %[[C4]] {
//         CHECK:  gpu.barrier
//         CHECK:  %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"} : () -> index
//         CHECK:  scf.for %[[IND:.+]] = %[[TIDX]] to %[[C2]] step %[[C2]] {
//     CHECK-DAG:    %[[SRC:.+]] = memref.subview %{{.*}}[%[[IND]], 0] [1, 4] [1, 1] : memref<2x4xf32, #{{.*}}> to memref<1x4xf32, #{{.*}}>
//     CHECK-DAG:    %[[DST:.+]] = memref.subview %{{.*}}[%[[IND]], 0] [1, 4] [1, 1] : memref<2x4xf32, #{{.*}}, 3> to memref<1x4xf32, #{{.*}}, 3>
//         CHECK:    linalg.copy(%[[SRC]], %[[DST]]) {__internal_linalg_transform__ = "vectorize"} : memref<1x4xf32, #{{.*}}>, memref<1x4xf32, #{{.*}}, 3>
//         CHECK:    }
//         CHECK:  %[[TIDX:.+]] = "gpu.thread_id"() {dimension = "x"} : () -> index
//         CHECK:  scf.for %[[IND0:.+]] = %[[C0]] to %[[C4]] step %[[C1]] {
//         CHECK:    scf.for %[[IND1:.+]] = %{{.*}} to %[[C256]] step %[[C256]] {
//     CHECK-DAG:      %[[SRC:.+]] = memref.subview %{{.*}}[%[[IND0]], %[[IND1]]] [1, 4] [1, 1] : memref<4x256xf32, #{{.*}}> to memref<1x4xf32, #{{.*}}>
//     CHECK-DAG:      %[[DST:.+]] = memref.subview %{{.*}}[%[[IND0]], %[[IND1]]] [1, 4] [1, 1] : memref<4x256xf32, #{{.*}}, 3> to memref<1x4xf32, #{{.*}}, 3>
//         CHECK:      linalg.copy(%[[SRC]], %[[DST]]) {__internal_linalg_transform__ = "vectorize"} : memref<1x4xf32, #{{.*}}>, memref<1x4xf32, #{{.*}}, 3>
//         CHECK:    }
//         CHECK:  }
//         CHECK:  gpu.barrier
//         CHECK:  scf.for %[[IND0:.+]] = %{{.*}} to %[[C2]] step %[[C2]] {
//         CHECK:    scf.for %[[IND1:.+]] = %{{.*}} to %[[C256]] step %[[C256]] {
//     CHECK-DAG:      %[[A:.+]] = memref.subview %17[%[[IND0]], 0] [2, 4] [1, 1] : memref<2x4xf32, #{{.*}}, 3> to memref<2x4xf32, #{{.*}}, 3>
//     CHECK-DAG:      %[[B:.+]] = memref.subview %18[0, %[[IND1]]] [4, 4] [1, 1] : memref<4x256xf32, #{{.*}}, 3> to memref<4x4xf32, #{{.*}}, 3>
//     CHECK-DAG:      %[[C:.+]] = memref.subview %11[%[[IND0]], %[[IND1]]] [2, 4] [1, 1] : memref<2x256xf32, #{{.*}}> to memref<2x4xf32, #{{.*}}>
//         CHECK:      linalg.matmul {__internal_linalg_transform__ = "vectorize", {{.*}}} ins(%[[A]], %[[B]] : memref<2x4xf32, #{{.*}}, 3>, memref<4x4xf32, #{{.*}}, 3>) outs(%[[C]] : memref<2x4xf32, #{{.*}}>)
//         CHECK:    }
//         CHECK:  }

// -----

// Pure reducion case, skip tiling.
hal.executable @reduction_dispatch {
hal.executable.variant @cuda, target="cuda" {
    hal.executable.entry_point @predict_dispatch_153 attributes {interface = @io, ordinal = 0 : index}
    module  {
      func @predict_dispatch_153() attributes {llvmgpu_workgroup_size = dense<[32, 1, 1]> : vector<3xi64>} {
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

//   CHECK-LABEL: hal.executable @reduction_dispatch
//         CHECK: linalg.fill(%{{.*}}, %{{.*}}) : f32, memref<f32>
//         CHECK: linalg.generic {{.*}} ins(%{{.*}} : memref<1000xf32>) outs(%{{.*}} : memref<f32>)
