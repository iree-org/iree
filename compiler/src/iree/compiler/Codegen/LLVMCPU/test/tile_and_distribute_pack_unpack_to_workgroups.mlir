// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-test-llvmcpu-materialize-dispatch-workgroup-count,iree-codegen-tile-and-distribute-to-workgroups)), canonicalize, cse)' --split-input-file %s | FileCheck %s
// TODO(hanchung): Drop iree-codegen-tile-and-distribute-to-workgroups pipeline
// from the test.

hal.executable private @pack_lowering {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @gemm_lhs_pack ordinal(0)
        layout(#hal.pipeline.layout<push_constants = 0,
            sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_set_encoding_op %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @gemm_lhs_pack() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64)
            : !flow.dispatch.tensor<readonly:tensor<100x250xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64)
            : !flow.dispatch.tensor<writeonly:tensor<14x64x8x4xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [100, 250], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<100x250xf32>> -> tensor<100x250xf32>
        %3 = tensor.empty() : tensor<14x64x8x4xf32>
        %4 = iree_linalg_ext.pack {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>} %2
            padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %3
            : (tensor<100x250xf32> tensor<14x64x8x4xf32>) -> tensor<14x64x8x4xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [14, 64, 8, 4], strides = [1, 1, 1, 1]
            : tensor<14x64x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<14x64x8x4xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 64)
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> ((s0 ceildiv 4) ceildiv 64)
//      CHECK: hal.executable.export public @gemm_lhs_pack
// CHECK-NEXT:   %[[ARG0:.+]]: !hal.device
// CHECK-SAME:   %[[ARG1:.+]]: index,
// CHECK-SAME:   %[[ARG2:.+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[W0:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//  CHECK-DAG:   %[[W1:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]]]
//      CHECK:   hal.return %[[W1]], %[[W0]], %[[C1]]

// -----

hal.executable private @pack_lowering {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @gemm_rhs_transpose_pack ordinal(0)
        layout(#hal.pipeline.layout<push_constants = 0,
            sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_set_encoding_op %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @gemm_rhs_transpose_pack() {
        %c0 = arith.constant 0 : index
        %c114688 = arith.constant 114688 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64)
            : !flow.dispatch.tensor<readonly:tensor<250x500xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c114688) alignment(64)
            : !flow.dispatch.tensor<writeonly:tensor<64x64x8x4xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<250x500xf32>> -> tensor<250x500xf32>
        %3 = tensor.empty() : tensor<64x64x8x4xf32>
        %4 = iree_linalg_ext.pack {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>} %2
            padding_value(%cst : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 4] into %3
            : (tensor<250x500xf32> tensor<64x64x8x4xf32>) -> tensor<64x64x8x4xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [64, 64, 8, 4], strides = [1, 1, 1, 1]
            : tensor<64x64x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x64x8x4xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 64)
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> ((s0 ceildiv 4) ceildiv 64)
//      CHECK: hal.executable.export public @gemm_rhs_transpose_pack
// CHECK-NEXT:   %[[ARG0:.+]]: !hal.device
// CHECK-SAME:   %[[ARG1:.+]]: index,
// CHECK-SAME:   %[[ARG2:.+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[W0:.+]] = affine.apply #[[MAP0]]()[%[[ARG2]]]
//  CHECK-DAG:   %[[W1:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
//      CHECK:   hal.return %[[W1]], %[[W0]], %[[C1]]

// -----

hal.executable private @clone_index_computations {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @clone_index_computations ordinal(0) layout(
        #hal.pipeline.layout<push_constants = 4, sets = [
            <0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_set_encoding_op %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @clone_index_computations() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%4, %5}
        %9 = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%6]
        %10 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%7]
        %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64)
            : !flow.dispatch.tensor<writeonly:tensor<?x?x8x4xf32>>{%9, %10}
        %12 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%4, %5} -> tensor<?x?xf32>
        %13 = affine.apply affine_map<()[s0, s1] -> (-s0 + s1 + (s0 ceildiv 16) * 16)>()[%4, %4]
        %14 = affine.apply affine_map<()[s0, s1] -> (-s0 + s1 + (s0 ceildiv 16) * 16)>()[%5, %5]
        %15 = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%13]
        %16 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%14]
        %17 = tensor.empty(%15, %16) : tensor<?x?x8x4xf32>
        %18 = iree_linalg_ext.pack {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>} %12
            padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %17
            : (tensor<?x?xf32> tensor<?x?x8x4xf32>) -> tensor<?x?x8x4xf32>
        %19 = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%6]
        %20 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%7]
        flow.dispatch.tensor.store %18, %11, offsets = [0, 0, 0, 0], sizes = [%19, %20, 8, 4], strides = [1, 1, 1, 1]
            : tensor<?x?x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x8x4xf32>>{%19, %20}
        return
      }
    }
  }
}
//   CHECK-DAG: #[[MAP7:.+]] = affine_map<(d0)[s0] -> (-d0 + (s0 ceildiv 16) * 2, 64)>
//   CHECK-DAG: #[[MAP8:.+]] = affine_map<(d0)[s0] -> (-d0 + (s0 ceildiv 16) * 4, 64)>
//       CHECK: func @clone_index_computations()
//       CHECK:   scf.for
//       CHECK:     %[[SIZE_Y:.+]] = affine.min #[[MAP7]](%{{.+}})[%{{.+}}]
//       CHECK:     scf.for
//       CHECK:       %[[SIZE_X:.+]] = affine.min #[[MAP8]](%{{.+}})[%{{.+}}]
//       CHECK:       flow.dispatch.tensor.store
//  CHECK-SAME:           sizes = [%[[SIZE_Y]], %[[SIZE_X]], 8, 4]

// -----

hal.executable private @dynamic_unpack {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @dynamic_unpack ordinal(0) layout(
        #hal.pipeline.layout<push_constants = 4, sets = [
            <0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg1]
      %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg2]
      hal.return %1, %0, %c1 : index, index, index
    }
    builtin.module {
      func.func @dynamic_unpack() {
        %c131072 = arith.constant 131072 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5}
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c131072) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        %10 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [%4, %5, 32, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5} -> tensor<?x?x32x16xi32>
        %11 = tensor.empty(%6, %7) : tensor<?x?xi32>
        %12 = iree_linalg_ext.unpack {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>}
          %10 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %11 : (tensor<?x?x32x16xi32> tensor<?x?xi32>) -> tensor<?x?xi32>
        flow.dispatch.tensor.store %12, %9, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        return
      }
    }
  }
}
// CHECK-LABEL: func.func @dynamic_unpack
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             iree_linalg_ext.unpack

// -----

hal.executable private @dynamic_unpack_dynamic_tile {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {}> {
    hal.executable.export public @dynamic_unpack_dynamic_tile ordinal(0) layout(
        #hal.pipeline.layout<push_constants = 4, sets = [
            <0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg1]
      %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg2]
      hal.return %1, %0, %c1 : index, index, index
    }
    builtin.module {
      func.func @dynamic_unpack_dynamic_tile() {
        %c131072 = arith.constant 131072 : index
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%4, %5, %c32, %c16}
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c131072) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        %10 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [%4, %5, %c32, %c16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%4, %5, %c32, %c16} -> tensor<?x?x?x?xi32>
        %11 = tensor.empty(%6, %7) : tensor<?x?xi32>
        %12 = iree_linalg_ext.unpack {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>}
          %10 inner_dims_pos = [0, 1] inner_tiles = [%c32, %c16] into %11 : (tensor<?x?x?x?xi32> tensor<?x?xi32>) -> tensor<?x?xi32>
        flow.dispatch.tensor.store %12, %9, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        return
      }
    }
  }
}
// CHECK-LABEL: func.func @dynamic_unpack_dynamic_tile
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             iree_linalg_ext.unpack

// -----

hal.executable private @unpack_elem {
  hal.executable.variant public @embedded_elf_arm_64, target = <"llvm-cpu", "embedded-elf-arm_64", {}> {
    hal.executable.export public @unpack_elem ordinal(0) layout(
        #hal.pipeline.layout<push_constants = 0, sets = [
            <0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg1]
      %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg2]
      hal.return %1, %0, %c1 : index, index, index
    }
    builtin.module {
      func.func @unpack_elem() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<16x48x8x8xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 48, 8, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x48x8x8xf32>> -> tensor<16x48x8x8xf32>
        %3 = tensor.empty() : tensor<128x384xf32>
        %4 = iree_linalg_ext.unpack {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>} %2 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %3 : (tensor<16x48x8x8xf32> tensor<128x384xf32>) -> tensor<128x384xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<128x384xf32>) outs(%3 : tensor<128x384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.addf %in, %in : f32
          linalg.yield %6 : f32
        } -> tensor<128x384xf32>
        flow.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : tensor<128x384xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @unpack_elem
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             iree_linalg_ext.unpack
// CHECK:             linalg.generic
