// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-cuda-tile-and-distribute))" %s | IreeFileCheck %s

hal.executable @add_dispatch_0 attributes {sym_visibility = "private"} {
hal.executable.target @cuda, filter="cuda" {
  hal.executable.entry_point @add_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : index, signature = (!flow.dispatch.tensor<readonly:1024xf32>, !flow.dispatch.tensor<readonly:1024xf32>, !flow.dispatch.tensor<writeonly:1024xf32>) -> ()}
  module  {
    func @add_dispatch_0() {
      %c0 = constant 0 : index
      %c1024 = constant 1024 : index
      %0 = hal.interface.binding.subspan @legacy_io::@ro0[%c0] : memref<1024xf32>
      %1 = hal.interface.binding.subspan @legacy_io::@ro1[%c0] : memref<1024xf32>
      %2 = hal.interface.binding.subspan @legacy_io::@wo2[%c0] : memref<1024xf32>
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
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%6, %7 : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>, memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>) outs(%8 : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
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
//  CHECK-SAME: cuda_workgroup_size = dense<[32, 1, 1]>
//      CHECK:    "gpu.thread_id"() {dimension = "x"}
//      CHECK:    scf.parallel
//      CHECK:      linalg.generic
