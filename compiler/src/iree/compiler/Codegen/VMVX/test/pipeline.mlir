// RUN: iree-opt  --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy, iree-llvmcpu-lower-executable-target)))" --split-input-file %s | FileCheck %s

hal.executable private @mmt4d_ukernel {
  hal.executable.variant public @vmvx_bytecode_fb target(<"vmvx", "vmvx-bytecode-fb", {ukernels = true}>) {
    hal.executable.export public @mmt4d_i8 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mmt4d_i8() {
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %c16 = arith.constant 16 : index
        %0:2 = iree_codegen.query_tile_sizes tensor<16x16xi8, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [i8, i8, i32]>> -> index, index
        %1 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%0#0]
        %2 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%0#1]
        %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%1, %2, %0#0, %0#1}
        %4:2 = iree_codegen.query_tile_sizes tensor<16x16xi8, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [i8, i8, i32]>> -> index, index
        %5 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%4#0]
        %6 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%4#1]
        %7 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c256) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%5, %6, %4#0, %4#1}
        %8:2 = iree_codegen.query_tile_sizes tensor<16x16xi32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [i8, i8, i32]>> -> index, index
        %9 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%8#0]
        %10 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%8#1]
        %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c512) : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%9, %10, %8#0, %8#1}
        %15 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [%1, %2, %0#0, %0#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%1, %2, %0#0, %0#1} -> tensor<?x?x?x?xi8>
        %19 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0, 0], sizes = [%5, %6, %4#0, %4#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%5, %6, %4#0, %4#1} -> tensor<?x?x?x?xi8>
        %23 = flow.dispatch.tensor.load %11, offsets = [0, 0, 0, 0], sizes = [%9, %10, %8#0, %8#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%9, %10, %8#0, %8#1} -> tensor<?x?x?x?xi32>
        %24 = linalg.mmt4d ins(%15, %19 : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>) outs(%23 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
        flow.dispatch.tensor.store %24, %11, offsets = [0, 0, 0, 0], sizes = [%9, %10, %8#0, %8#1], strides = [1, 1, 1, 1] : tensor<?x?x?x?xi32> -> !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%9, %10, %8#0, %8#1}
        return
      }
    }
  }
}
// CHECK: func private @vmvx.mmt4d(
// CHECK: func @mmt4d_i8()
// CHECK:   func.call @vmvx.mmt4d(
