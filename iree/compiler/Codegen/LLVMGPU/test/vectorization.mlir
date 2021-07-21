// RUN: iree-opt -iree-llvmgpu-vectorization %s | IreeFileCheck %s

func @add_dispatch_0() attributes {cuda_workgroup_size = dense<[32, 1, 1]> : vector<3xi64>} {
  %c128 = constant 128 : index
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c1024 = constant 1024 : index
  %0 = hal.interface.binding.subspan @io::@ro0[%c0] : memref<1024x1024x1024xf32>
  %1 = hal.interface.binding.subspan @io::@ro1[%c0] : memref<1024x1024x1024xf32>
  %2 = hal.interface.binding.subspan @io::@wo2[%c0] : memref<1024x1024x1024xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_count_z = hal.interface.workgroup.count[2] : index
  scf.for %arg0 = %workgroup_id_z to %c1024 step %workgroup_count_z {
    scf.for %arg1 = %workgroup_id_y to %c1024 step %workgroup_count_y {
      %3 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
      %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
      scf.for %arg2 = %3 to %c1024 step %4 {
        %5 = memref.subview %0[%arg0, %arg1, %arg2] [1, 1, 128] [1, 1, 1] : memref<1024x1024x1024xf32> to memref<1x1x128xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
        %6 = memref.subview %1[%arg0, %arg1, %arg2] [1, 1, 128] [1, 1, 1] : memref<1024x1024x1024xf32> to memref<1x1x128xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
        %7 = memref.subview %2[%arg0, %arg1, %arg2] [1, 1, 128] [1, 1, 1] : memref<1024x1024x1024xf32> to memref<1x1x128xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
        %8 = "gpu.thread_id"() {dimension = "x"} : () -> index
        %9 = "gpu.block_dim"() {dimension = "x"} : () -> index
        %10 = "gpu.thread_id"() {dimension = "y"} : () -> index
        %11 = "gpu.block_dim"() {dimension = "y"} : () -> index
        %12 = "gpu.thread_id"() {dimension = "z"} : () -> index
        %13 = "gpu.block_dim"() {dimension = "z"} : () -> index
        scf.for %arg3 = %12 to %c1 step %13 {
          scf.for %arg4 = %10 to %c1 step %11 {
            %14 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%8]
            %15 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%9]
            scf.for %arg5 = %14 to %c128 step %15 {
              %16 = memref.subview %5[%arg3, %arg4, %arg5] [1, 1, 4] [1, 1, 1] : memref<1x1x128xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>> to memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              %17 = memref.subview %6[%arg3, %arg4, %arg5] [1, 1, 4] [1, 1, 1] : memref<1x1x128xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>> to memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              %18 = memref.subview %7[%arg3, %arg4, %arg5] [1, 1, 4] [1, 1, 1] : memref<1x1x128xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>> to memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16, %17 : memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>, memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>) outs(%18 : memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 1048576 + s0 + d1 * 1024 + d2)>>) attrs =  {__internal_linalg_transform__ = "vectorize", is_root_op, launch_info_key = "__op_num_0__"} {
              ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):  // no predecessors
                %19 = addf %arg6, %arg7 : f32
                linalg.yield %19 : f32
              }
            }
          }
        }
      }
    }
  }
  return
}
// CHECK-LABEL: func @add_dispatch_0()
//       CHECK:   vector.transfer_read {{.*}} : memref<1024x1024x1024xf32>, vector<4xf32>
//       CHECK:   vector.transfer_read {{.*}} : memref<1024x1024x1024xf32>, vector<4xf32>
//       CHECK:   addf %{{.*}}, %{{.*}}  : vector<1x1x4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, memref<1024x1024x1024xf32>
