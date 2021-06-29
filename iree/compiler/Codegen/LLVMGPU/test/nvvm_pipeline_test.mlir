// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.variant(iree-codegen-linalg-to-nvvm-pipeline))" %s | IreeFileCheck %s

// Verify that a simple element wise op gets lowered succefully all the way to
// nvvm/llvm dialect.

hal.executable @simpleMath_ex_dispatch_0 {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @cuda, filter="cuda" {
  hal.executable.entry_point @add_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
  module  {
    func @add_dispatch_0() {
      %c0 = constant 0 : index
      %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:16xf32>
      %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:16xf32>
      %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:16xf32>
      %3 = linalg.init_tensor [16] : tensor<16xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %5 = flow.dispatch.tensor.load %1, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16xf32>
        flow.dispatch.tensor.store %6, %2, offsets=[], sizes=[], strides=[] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
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

// CHECK-LABEL: hal.executable @simpleMath_ex_dispatch_0
//       CHECK:   hal.executable.variant @cuda, filter="cuda" {
//       CHECK:   llvm.fadd

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
hal.executable @dot_dispatch_0 attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @cuda, filter="cuda" {
    hal.executable.entry_point @dot_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
    module  {
      func @dot_dispatch_0() {
        %cst = constant 0.000000e+00 : f32
        %c0 = constant 0 : index
        %c1024 = constant 1024 : index
        %c1 = constant 1 : index
        %0 = hal.interface.binding.subspan @io::@ro0[%c0] : !flow.dispatch.tensor<readonly:1024x1024xf32>
        %1 = hal.interface.binding.subspan @io::@ro1[%c0] : !flow.dispatch.tensor<readonly:1024x1024xf32>
        %2 = hal.interface.binding.subspan @io::@wo2[%c0] : !flow.dispatch.tensor<writeonly:1024x1024xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c1024 step %4 {
          %5 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c1024 step %6 {
            %7 = affine.min #map1(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [%7, %c1024], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:1024x1024xf32> -> tensor<?x1024xf32>
            %9 = affine.min #map1(%arg1)[%workgroup_size_x]
            %10 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c1024, %9], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:1024x1024xf32> -> tensor<1024x?xf32>
            %11 = affine.min #map1(%arg0)[%workgroup_size_y]
            %12 = affine.min #map1(%arg1)[%workgroup_size_x]
            %13 = affine.min #map2(%arg0)[%workgroup_size_y]
            %14 = affine.min #map2(%arg1)[%workgroup_size_x]
            %15 = linalg.init_tensor [%13, %14] : tensor<?x?xf32>
            %16 = linalg.fill(%cst, %15) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
            %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%8, %10 : tensor<?x1024xf32>, tensor<1024x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
            flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:1024x1024xf32>
          }
        }
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//   CHECK-LABEL: hal.executable @dot_dispatch_0
//         CHECK:   hal.executable.variant @cuda, filter="cuda" {
// CHECK-COUNT-2:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
//         CHECK:   llvm.br
// CHECK-COUNT-6:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>, 3>
// CHECK-COUNT-8:   "llvm.intr.fmuladd"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//         CHECK:   llvm.br
// CHECK-COUNT-2:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>>

// -----

hal.executable @conv2d_dispatch_0 attributes {sym_visibility = "private"} {
hal.executable.variant @cuda, filter="cuda" {
  hal.interface @io {
    hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.entry_point @conv2d_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
  module  {
    func @conv2d_dispatch_0() {
      %c0 = constant 0 : index
      %cst = constant 0.000000e+00 : f32
      %c2 = constant 2 : index
      %c3 = constant 3 : index
      %c1 = constant 1 : index
      %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:1x4x4x2xf32>
      %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:3x2x2x1xf32>
      %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:1x2x3x1xf32>
      %workgroup_size_x = hal.interface.workgroup.size[0] : index
      %workgroup_size_y = hal.interface.workgroup.size[1] : index
      %workgroup_size_z = hal.interface.workgroup.size[2] : index
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %workgroup_count_y = hal.interface.workgroup.count[1] : index
      %workgroup_id_z = hal.interface.workgroup.id[2] : index
      %workgroup_count_z = hal.interface.workgroup.count[2] : index
      %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
      %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
      scf.for %arg0 = %3 to %c2 step %4 {
        %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg1 = %5 to %c3 step %6 {
          %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg2 = %7 to %c1 step %8 {
            %9 = affine.min affine_map<(d0)[s0] -> (s0 + 2, -d0 + 4)>(%arg0)[%workgroup_size_z]
            %10 = affine.min affine_map<(d0)[s0] -> (s0 + 1, -d0 + 4)>(%arg1)[%workgroup_size_y]
            %11 = flow.dispatch.tensor.load %0, offsets = [0, %arg0, %arg1, 0], sizes = [1, %9, %10, 2], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x4x4x2xf32> -> tensor<1x?x?x2xf32>
            %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg2)[%workgroup_size_x]
            %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 2, 2, %12], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x2x2x1xf32> -> tensor<3x2x2x?xf32>
            %14 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 2)>(%arg0)[%workgroup_size_z]
            %15 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 3)>(%arg1)[%workgroup_size_y]
            %16 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg2)[%workgroup_size_x]
            %17 = affine.min affine_map<(d0)[s0] -> (-d0 + 2, s0)>(%arg0)[%workgroup_size_z]
            %18 = affine.min affine_map<(d0)[s0] -> (-d0 + 3, s0)>(%arg1)[%workgroup_size_y]
            %19 = affine.min affine_map<(d0)[s0] -> (-d0 + 1, s0)>(%arg2)[%workgroup_size_x]
            %20 = linalg.init_tensor [1, %17, %18, %19] : tensor<1x?x?x?xf32>
            %21 = linalg.fill(%cst, %20) : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
            %22 = linalg.conv_2d_input_nhwc_filter_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%11, %13 : tensor<1x?x?x2xf32>, tensor<3x2x2x?xf32>) outs(%21 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
            flow.dispatch.tensor.store %22, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %14, %15, %16], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x2x3x1xf32>
          }
        }
      }
      return
    }
    hal.interface @io attributes {sym_visibility = "private"} {
      hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
  }
}
}

//   CHECK-LABEL: hal.executable @conv2d_dispatch_0
//         CHECK:   hal.executable.variant @cuda, filter="cuda" {
// CHECK-COUNT-3:   llvm.load %{{.*}} : !llvm.ptr<f32>
//         CHECK:   lvm.fmul %{{.*}}, %{{.*}}  : f32
//         CHECK:   llvm.fadd %{{.*}}, %{{.*}}  : f32
//         CHECK:   llvm.store {{.*}} : !llvm.ptr<f32>

// -----

hal.executable @simpleMath_ex_dispatch_0 {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @cuda, filter="cuda" {
  hal.executable.entry_point @add_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
  module  {
    func @add_dispatch_0() {
      %c0 = constant 0 : index
      %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:16xf32>
      %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:16xf32>
      %3 = linalg.init_tensor [16] : tensor<16xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %5 = constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16xf32>
        flow.dispatch.tensor.store %6, %2, offsets=[], sizes=[], strides=[] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// CHECK-LABEL: hal.executable @simpleMath_ex_dispatch_0
//       CHECK:   hal.executable.variant @cuda, filter="cuda" {
//       CHECK:   llvm.mlir.global private constant @{{.*}}(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<16xf32>)
//       CHECK:   llvm.fadd

// -----

hal.executable @reduction_dispatch {
hal.executable.variant @cuda, filter="cuda" {
  hal.executable.entry_point @reduction attributes {interface = @io, ordinal = 0 : index}
  module  {
    func @reduction() {
      %c0 = constant 0 : index
      %cst = constant 0.000000e+00 : f32
      %c96 = constant 96 : index
      %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:14x14x96xf32>
      %1 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : !flow.dispatch.tensor<writeonly:96xf32>
      %workgroup_size_x = hal.interface.workgroup.size[0] : index
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      %2 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
      %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
      scf.for %arg0 = %2 to %c96 step %3 {
        %4 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 96)>(%arg0)[%workgroup_size_x]
        %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, %arg0], sizes = [14, 14, %4], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:14x14x96xf32> -> tensor<14x14x?xf32>
        %6 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 96)>(%arg0)[%workgroup_size_x]
        %7 = affine.min affine_map<(d0)[s0] -> (-d0 + 96, s0)>(%arg0)[%workgroup_size_x]
        %8 = linalg.init_tensor [%7] : tensor<?xf32>
        %9 = linalg.fill(%cst, %8) : f32, tensor<?xf32> -> tensor<?xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%5 : tensor<14x14x?xf32>) outs(%9 : tensor<?xf32>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
        ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
          %11 = addf %arg1, %arg2 : f32
          linalg.yield %11 : f32
        } -> tensor<?xf32>
        flow.dispatch.tensor.store %10, %1, offsets = [%arg0], sizes = [%6], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:96xf32>
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

// CHECK-LABEL: hal.executable @reduction_dispatch
//       CHECK:   hal.executable.variant @cuda, filter="cuda" {
//       CHECK:   llvm.fadd
