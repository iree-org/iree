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
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:16384xf32>
      %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:16384xf32>
      %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:16384xf32>
      %3 = linalg.init_tensor [16384] : tensor<16384xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16384xf32> -> tensor<16384xf32>
      %5 = flow.dispatch.tensor.load %1, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16384xf32> -> tensor<16384xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16384xf32>, tensor<16384xf32>) outs(%3 : tensor<16384xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = arith.addf %arg0, %arg1 : f32
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

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[256]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 256)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUVectorize", workload_per_wg = [256]>
//      CHECK: hal.executable.entry_point public @add_dispatch_0
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [64 : index, 1 : index, 1 : index]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index,
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
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
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant 0.000000e+00 : f32
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
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[4, 2, 4]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = [2, 4]>
//      CHECK: hal.executable.entry_point public @dot_dispatch_1
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [2 : index, 4 : index, 1 : index]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index,
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
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
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0x7FC00000 : f32
        %cst_0 = arith.constant 0xFF800000 : f32
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<1000xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : memref<f32>
        linalg.fill(%cst_0, %1) : f32, memref<f32>
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%0 : memref<1000xf32>) outs(%1 : memref<f32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %2 = arith.cmpf ogt, %arg0, %arg1 : f32
          %3 = select %2, %arg0, %arg1 : f32
          %4 = arith.cmpf uno, %arg0, %arg1 : f32
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

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUDistribute", workload_per_wg = []>
//      CHECK: hal.executable.entry_point public @predict_dispatch_153
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [1 : index, 1 : index, 1 : index]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index,
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//      CHECK:     hal.return %[[C1]], %[[C1]], %[[C1]]
//      CHECK: linalg.fill
// CHECK-SAME:   lowering.config = #[[CONFIG]]
//      CHECK: linalg.generic
// CHECK-SAME:   lowering.config = #[[CONFIG]]

// -----

hal.executable @tensor_insert {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @tensor_insert_slice attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      builtin.func @tensor_insert_slice() {
        %c0 = arith.constant 0 : index
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
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 128)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUDistribute", workload_per_wg = [128, 1]>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[NWGSX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//      CHECK:   hal.return %[[NWGSX]], %[[ARG1]], %[[C1]]

// -----

hal.executable @tensor_insert {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @tensor_insert_slice attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      builtin.func @tensor_insert_slice() {
        %c0 = arith.constant 0 : index
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
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 256]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 256)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUVectorize", workload_per_wg = [256, 1]>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[NWGSX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//      CHECK:   hal.return %[[NWGSX]], %[[ARG1]], %[[C1]]
//      CHECK:   linalg.copy
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

hal.executable private @static_1d_fft_stage2  {
  hal.interface @io {
    hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
    hal.interface.binding @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @static_1d_fft_stage2 attributes {interface = @io, ordinal = 0 : index}
    builtin.module {
      builtin.func @static_1d_fft_stage2() {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
        %0 = hal.interface.binding.subspan @io::@s0b0_rw_external[%c0] : !flow.dispatch.tensor<readwrite:32xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_rw_external[%c0] : !flow.dispatch.tensor<readwrite:32xf32>
        %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:32xf32> -> tensor<32xf32>
        %3 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:32xf32> -> tensor<32xf32>
        %4:2 = linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
        flow.dispatch.tensor.store %4#0, %0, offsets = [], sizes = [], strides = [] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:32xf32>
        flow.dispatch.tensor.store %4#1, %1, offsets = [], sizes = [], strides = [] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:32xf32>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[4]{{\]}}, native_vector_size = []>
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUDistribute", workload_per_wg = [4]>
//       CHECK: hal.executable.entry_point public @static_1d_fft_stage2
//  CHECK-SAME:   translation.info = #[[TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [32 : index, 1 : index, 1 : index]
//  CHECK-NEXT: ^{{.+}}(%[[ARG0:.+]]: index, %{{.+}}: index, %{{.+}}: index):
//  CHECK-NEXT:   %[[ONE:.+]] = arith.constant 1 : index
//  CHECK-NEXT:   %[[T:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-NEXT:   hal.return %[[T]], %[[ONE]], %[[ONE]]

//       CHECK: func @static_1d_fft_stage2()
//       CHECK:   linalg_ext.fft
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----


hal.executable private @static_3d_fft_stage3  {
  hal.interface @io {
    hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
    hal.interface.binding @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @static_3d_fft_stage3 attributes {interface = @io, ordinal = 0 : index}
    builtin.module {
      builtin.func @static_3d_fft_stage3() {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<[1.000000e+00, 0.707106769, 6.12323426E-17, -0.707106769]> : tensor<4xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -0.707106769, -1.000000e+00, -0.707106769]> : tensor<4xf32>
        %0 = memref.buffer_cast %cst_0 : memref<4xf32>
        %1 = memref.buffer_cast %cst : memref<4xf32>
        %2 = hal.interface.binding.subspan @io::@s0b0_rw_external[%c0] : memref<64x128x32xf32>
        %3 = hal.interface.binding.subspan @io::@s0b1_rw_external[%c0] : memref<64x128x32xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c64 step %workgroup_count_z {
          scf.for %arg1 = %workgroup_id_y to %c128 step %workgroup_count_y {
            %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_x]
            %5 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_x]
            scf.for %arg2 = %4 to %c32 step %5 {
              %6 = memref.subview %2[%arg0, %arg1, %arg2] [1, 1, 4] [1, 1, 1] : memref<64x128x32xf32> to memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              %7 = memref.cast %6 : memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              %8 = memref.subview %3[%arg0, %arg1, %arg2] [1, 1, 4] [1, 1, 1] : memref<64x128x32xf32> to memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              %9 = memref.cast %8 : memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              linalg_ext.fft {__internal_linalg_transform__ = "workgroup"}
                ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>)
                outs(%7, %9 : memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>, memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>)
            }
          }
        }
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 1, 8]{{\]}}, native_vector_size = []>
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUDistribute", workload_per_wg = [8, 1, 1]>
//       CHECK: hal.executable.entry_point public @static_3d_fft_stage3
//  CHECK-SAME:   translation.info = #[[TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [32 : index, 1 : index, 1 : index]
//  CHECK-NEXT: ^{{.+}}(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index):
//  CHECK-NEXT:   %[[T:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-NEXT:   hal.return %[[T]], %[[ARG1]], %[[ARG2]]

//       CHECK: func @static_3d_fft_stage3()
//       CHECK:   linalg_ext.fft
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#compilation = #iree_codegen.compilation.info<
    #iree_codegen.lowering.config<tile_sizes = [[32, 256, 64]], native_vector_size = []>,
    #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = [256, 32]>,
    workgroup_size = [16, 8, 1]>
hal.executable @user_config {
hal.executable.variant public @cuda_nvptx_fb, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point public @_lowering_config_test_dispatch_1 attributes {interface = @io, ordinal = 0 : index}
  builtin.module  {
    func @_lowering_config_test_dispatch_1() {
      %cst = arith.constant 0.000000e+00 : f32
      %c128 = arith.constant 128 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:128x256xf32>
      %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:256x1024xf32>
      %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:128x1024xf32>
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
        scf.for %arg1 = %5 to %c1024 step %6 {
          %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 128)>(%arg0)[%workgroup_size_y]
          %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<?x256xf32>
          %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg1)[%workgroup_size_x]
          %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [256, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x1024xf32> -> tensor<256x?xf32>
          %11 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 128)>(%arg0)[%workgroup_size_y]
          %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1024)>(%arg1)[%workgroup_size_x]
          %13 = affine.min affine_map<(d0)[s0] -> (-d0 + 128, s0)>(%arg0)[%workgroup_size_y]
          %14 = affine.min affine_map<(d0)[s0] -> (-d0 + 1024, s0)>(%arg1)[%workgroup_size_x]
          %15 = linalg.init_tensor [%13, %14] : tensor<?x?xf32>
          %16 = linalg.fill(%cst, %15) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
          %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup", compilation.info = #compilation} ins(%8, %10 : tensor<?x256xf32>, tensor<256x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
          flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:128x1024xf32>
        }
      }
      return
    }
    hal.interface private @io {
      hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
  }
}
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[32, 256, 64]{{\]}}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = [256, 32]>
//      CHECK: hal.executable.entry_point public @_lowering_config_test_dispatch_1
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [16 : index, 8 : index, 1 : index]
//      CHECK: func @_lowering_config_test_dispatch_1
//      CHECK:   linalg.fill
// CHECK-SAME:       lowering.config = #[[CONFIG]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering.config = #[[CONFIG]]
