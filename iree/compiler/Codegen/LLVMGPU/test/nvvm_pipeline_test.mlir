// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-codegen-linalg-to-nvvm-pipeline))' %s | IreeFileCheck %s

// Verify that a simple element wise op gets lowered succefully all the way to
// nvvm/llvm dialect.

hal.executable @simpleMath_ex_dispatch_0 {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer"
  }
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @add_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
  builtin.module  {
    func @add_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:16xf32>
      %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:16xf32>
      %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:16xf32>
      %3 = linalg.init_tensor [16] : tensor<16xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %5 = flow.dispatch.tensor.load %1, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16xf32>
        flow.dispatch.tensor.store %6, %2, offsets=[], sizes=[], strides=[] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
        return
      }
      hal.interface private @io  {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @simpleMath_ex_dispatch_0
//       CHECK:   hal.executable.variant public @cuda
//       CHECK:   llvm.fadd

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
hal.executable @dot_dispatch_0 {
  hal.interface @io {
    hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @dot_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @dot_dispatch_0() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
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
            %17 = linalg.matmul ins(%8, %10 : tensor<?x1024xf32>, tensor<1024x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
            flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:1024x1024xf32>
          }
        }
        return
      }
      hal.interface private @io  {
        hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}

//     CHECK-LABEL: hal.executable public @dot_dispatch_0
//           CHECK:   hal.executable.variant public @cuda
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-3:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
//           CHECK:   llvm.br
//   CHECK-COUNT-3:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//  CHECK-COUNT-32:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>, 3>
// CHECK-COUNT-128:   "llvm.intr.fmuladd"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//   CHECK-COUNT-3:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
//           CHECK:   llvm.br
//   CHECK-COUNT-3:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//  CHECK-COUNT-32:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>, 3>
// CHECK-COUNT-128:   "llvm.intr.fmuladd"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//   CHECK-COUNT-4:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>>

// -----

// Check that a generic op representing a matmul is getting the same
// configuration as the matmul op.
#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>

#matmul_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

hal.executable @dot_dispatch_0 {
  hal.interface @io {
    hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @dot_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @dot_dispatch_0() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
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
            %17 = linalg.generic #matmul_trait ins(%8, %10 : tensor<?x1024xf32>, tensor<1024x?xf32>) outs(%16 : tensor<?x?xf32>)  {
              ^bb(%a: f32, %b: f32, %c: f32) :
              %d = arith.mulf %a, %b: f32
              %e = arith.addf %c, %d: f32
              linalg.yield %e : f32
            } -> (tensor<?x?xf32>)
            flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:1024x1024xf32>
          }
        }
        return
      }
      hal.interface private @io  {
        hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}

//   CHECK-LABEL: hal.executable public @dot_dispatch_0
//         CHECK:   hal.executable.variant public @cuda
//         CHECK:   llvm.br
// CHECK-COUNT-8:   "llvm.intr.fmuladd"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//         CHECK:   llvm.br
// CHECK-COUNT-2:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>>

// -----

hal.executable @conv2d_dispatch_0 {
hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
  hal.interface @io {
    hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.entry_point @conv2d_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
  builtin.module  {
    func @conv2d_dispatch_0() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c1 = arith.constant 1 : index
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
            %22 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%11, %13 : tensor<1x?x?x2xf32>, tensor<3x2x2x?xf32>) outs(%21 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
            flow.dispatch.tensor.store %22, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %14, %15, %16], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x2x3x1xf32>
          }
        }
      }
      return
    }
    hal.interface private @io  {
      hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
      hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
      hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
    }
  }
}
}

//   CHECK-LABEL: hal.executable public @conv2d_dispatch_0
//         CHECK:   hal.executable.variant public @cuda
// CHECK-COUNT-3:   llvm.load %{{.*}} : !llvm.ptr<f32>
//         CHECK:   lvm.fmul %{{.*}}, %{{.*}}  : f32
//         CHECK:   llvm.fadd %{{.*}}, %{{.*}}  : f32
//         CHECK:   llvm.store {{.*}} : !llvm.ptr<f32>

// -----

hal.executable @simpleMath_ex_dispatch_0 {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer"
  }
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @add_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
  builtin.module  {
    func @add_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:16xf32>
      %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:16xf32>
      %3 = linalg.init_tensor [16] : tensor<16xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %5 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16xf32>
        flow.dispatch.tensor.store %6, %2, offsets=[], sizes=[], strides=[] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
        return
      }
      hal.interface private @io  {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer"
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @simpleMath_ex_dispatch_0
//       CHECK:   hal.executable.variant public @cuda
//       CHECK:   llvm.mlir.global private constant @{{.*}}(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<16xf32>)
//       CHECK:   llvm.fadd

// -----

hal.executable @reduction_dispatch {
hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @reduction attributes {interface = @io, ordinal = 0 : index}
  builtin.module  {
    func @reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c96 = arith.constant 96 : index
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
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%5 : tensor<14x14x?xf32>) outs(%9 : tensor<?xf32>) {
        ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
          %11 = arith.addf %arg1, %arg2 : f32
          linalg.yield %11 : f32
        } -> tensor<?xf32>
        flow.dispatch.tensor.store %10, %1, offsets = [%arg0], sizes = [%6], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:96xf32>
      }
      return
    }
    hal.interface private @io  {
      hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
      hal.interface.binding @s0b1_xw_external, set=0, binding=1, type="StorageBuffer"
    }
  }
}
}

// CHECK-LABEL: hal.executable public @reduction_dispatch
//       CHECK:   hal.executable.variant public @cuda
//       CHECK:   llvm.fadd

// -----

hal.executable @vector_add_dispatch {
hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @vector_add_dispatch attributes {interface = @io, ordinal = 0 : index}
  builtin.module  {
    builtin.func @vector_add_dispatch() {
      %c0 = arith.constant 0 : index
      %c16384 = arith.constant 16384 : index
      %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:16384xf32>
      %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:16384xf32>
      %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:16384xf32>
      %workgroup_size_x = hal.interface.workgroup.size[0] : index
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
      %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
      scf.for %arg0 = %3 to %c16384 step %4 {
        %5 = affine.min affine_map<(d0, d1) -> (d1, -d0 + 16384)>(%arg0)[%workgroup_size_x]
        %6 = flow.dispatch.tensor.load %0, offsets = [%arg0], sizes = [%5], strides = [1] : !flow.dispatch.tensor<readonly:16384xf32> -> tensor<?xf32>
        %7 = affine.min affine_map<(d0, d1) -> (d1, -d0 + 16384)>(%arg0)[%workgroup_size_x]
        %8 = flow.dispatch.tensor.load %1, offsets = [%arg0], sizes = [%7], strides = [1] : !flow.dispatch.tensor<readonly:16384xf32> -> tensor<?xf32>
        %9 = affine.min affine_map<(d0, d1) -> (d1, -d0 + 16384)>(%arg0)[%workgroup_size_x]
        %10 = linalg.init_tensor [%9] : tensor<?xf32>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%6, %8 : tensor<?xf32>, tensor<?xf32>) outs(%10 : tensor<?xf32>) {
        ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
          %12 = arith.addf %arg1, %arg2 : f32
          linalg.yield %12 : f32
        } -> tensor<?xf32>
        flow.dispatch.tensor.store %11, %2, offsets = [%arg0], sizes = [%9], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:16384xf32>
      }
      return
    }
    hal.interface private @io  {
      hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
      hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
       hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
    }
  }
}
}

//   CHECK-LABEL: hal.executable public @vector_add_dispatch
//         CHECK:   hal.executable.variant public @cuda
//         CHECK:   llvm.fadd %{{.*}}, %{{.*}}  : vector<4xf32
//         CHECK:   llvm.store %{{.*}} : !llvm.ptr<vector<4xf32>>

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 16384)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 16384, s0)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> (d0)>

hal.executable @vector_reduction_dispatch {
hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @vector_reduction_dispatch attributes {interface = @io, ordinal = 0 : index}
  builtin.module  {
    builtin.func @vector_reduction_dispatch() {
          %c0 = arith.constant 0 : index
          %c16384 = arith.constant 16384 : index
          %cst = arith.constant 1.000000e+00 : f32
          %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:512x16384xf32>
          %1 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : !flow.dispatch.tensor<writeonly:16384xf32>
          %workgroup_size_x = hal.interface.workgroup.size[0] : index
          %workgroup_id_x = hal.interface.workgroup.id[0] : index
          %workgroup_count_x = hal.interface.workgroup.count[0] : index
          %2 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
          %3 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg0 = %2 to %c16384 step %3 {
            %4 = affine.min #map1(%arg0)[%workgroup_size_x]
            %5 = flow.dispatch.tensor.load %0, offsets = [0, %arg0], sizes = [512, %4], strides = [1, 1] : !flow.dispatch.tensor<readonly:512x16384xf32> -> tensor<512x?xf32>
            %6 = affine.min #map1(%arg0)[%workgroup_size_x]
            %7 = affine.min #map2(%arg0)[%workgroup_size_x]
            %8 = linalg.init_tensor [%7] : tensor<?xf32>
            %9 = linalg.fill(%cst, %8) : f32, tensor<?xf32> -> tensor<?xf32>
            %10 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<512x?xf32>) outs(%9 : tensor<?xf32>) {
            ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
              %11 = arith.addf %arg1, %arg2 : f32
              linalg.yield %11 : f32
            } -> tensor<?xf32>
            flow.dispatch.tensor.store %10, %1, offsets = [%arg0], sizes = [%6], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:16384xf32>
          }
          return
        }
        hal.interface private @io  {
          hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
          hal.interface.binding @s0b1_xw_external, set=0, binding=1, type="StorageBuffer"
        }
    }
  }
}

//   CHECK-LABEL: hal.executable public @vector_reduction_dispatch
//         CHECK:   hal.executable.variant public @cuda
//         CHECK:   llvm.fadd %{{.*}}, %{{.*}} : vector<4xf32>
//         CHECK:   llvm.store %{{.*}} : !llvm.ptr<vector<4xf32>>

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
hal.executable @mma_fused {
  hal.interface @io {
    hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
    hal.executable.entry_point @mma_fused attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @mma_fused() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
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
            %17 = linalg.matmul ins(%8, %10 : tensor<?x1024xf32>, tensor<1024x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
              iterator_types = ["parallel", "parallel"]} ins(%17, %17 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
            ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
              %19 = arith.addf %arg3, %arg4 : f32
             linalg.yield %19 : f32
            } -> tensor<?x?xf32>
            flow.dispatch.tensor.store %18, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:1024x1024xf32>
          }
        }
        return
      }
      hal.interface private @io  {
        hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}

//     CHECK-LABEL: hal.executable public @mma_fused
//           CHECK:   hal.executable.variant public @cuda
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-2:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
//           CHECK:   llvm.br
//   CHECK-COUNT-2:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//   CHECK-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECK-COUNT-2:   nvvm.wmma.mma
//   CHECK-COUNT-2:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
//           CHECK:   llvm.br
//   CHECK-COUNT-2:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//   CHECK-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECK-COUNT-2:   nvvm.wmma.mma
//   CHECK-COUNT-8:   llvm.fadd
//   CHECK-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<f32>, f32, f32, f32, f32, f32, f32, f32, f32
