// RUN: iree-opt  --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))" --split-input-file %s | FileCheck %s

hal.executable private @mmt4d_ukernel {
  hal.executable.variant public @vmvx_bytecode_fb, target = <"vmvx", "vmvx-bytecode-fb", {ukernels = true}> {
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
        %0:2 = vmvx.query_tile_sizes sizes(%c16, %c16) flags(1048576) -> index, index
        %1 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%0#0]
        %2 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%0#1]
        %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%1, %2, %0#0, %0#1}
        %4:2 = vmvx.query_tile_sizes sizes(%c16, %c16) flags(1114112) -> index, index
        %5 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%4#0]
        %6 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%4#1]
        %7 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c256) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%5, %6, %4#0, %4#1}
        %8:2 = vmvx.query_tile_sizes sizes(%c16, %c16) flags(1179648) -> index, index
        %9 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%8#0]
        %10 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%8#1]
        %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c512) : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%9, %10, %8#0, %8#1}
        %12:2 = vmvx.query_tile_sizes sizes(%c16, %c16) flags(1048576) -> index, index
        %13 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%12#0]
        %14 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%12#1]
        %15 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [%13, %14, %12#0, %12#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%13, %14, %12#0, %12#1} -> tensor<?x?x?x?xi8>
        %16:2 = vmvx.query_tile_sizes sizes(%c16, %c16) flags(1114112) -> index, index
        %17 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%16#0]
        %18 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%16#1]
        %19 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0, 0], sizes = [%17, %18, %16#0, %16#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%17, %18, %16#0, %16#1} -> tensor<?x?x?x?xi8>
        %20:2 = vmvx.query_tile_sizes sizes(%c16, %c16) flags(1179648) -> index, index
        %21 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%20#0]
        %22 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%20#1]
        %23 = flow.dispatch.tensor.load %11, offsets = [0, 0, 0, 0], sizes = [%21, %22, %20#0, %20#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%21, %22, %20#0, %20#1} -> tensor<?x?x?x?xi32>
        %24 = linalg.mmt4d ins(%15, %19 : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>) outs(%23 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
        %25:2 = vmvx.query_tile_sizes sizes(%c16, %c16) flags(1179648) -> index, index
        %26 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%25#0]
        %27 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%25#1]
        flow.dispatch.tensor.store %24, %11, offsets = [0, 0, 0, 0], sizes = [%26, %27, %25#0, %25#1], strides = [1, 1, 1, 1] : tensor<?x?x?x?xi32> -> !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%26, %27, %25#0, %25#1}
        return
      }
    }
  }
}
// CHECK: func private @vmvx.mmt4d.i8i8i32(
// CHECK: func @mmt4d_i8()
// CHECK:   func.call @vmvx.mmt4d.i8i8i32(

// -----

hal.executable private @matmul_ukernel {
  hal.executable.variant public @vmvx_bytecode_fb, target = <"vmvx", "vmvx-bytecode-fb", {ukernels = true}> {
    hal.executable.export public @matmul_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 6, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_f32() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = hal.interface.constant.load[4] : i32
        %5 = hal.interface.constant.load[5] : i32
        %6 = arith.index_castui %0 : i32 to index
        %7 = arith.index_castui %1 : i32 to index
        %8 = arith.index_castui %2 : i32 to index
        %9 = arith.index_castui %3 : i32 to index
        %10 = arith.index_castui %4 : i32 to index
        %11 = arith.index_castui %5 : i32 to index
        %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %13 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%8, %9}
        %14 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%10, %11}
        %15 = flow.dispatch.tensor.load %12, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %16 = flow.dispatch.tensor.load %13, offsets = [0, 0], sizes = [%8, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%8, %9} -> tensor<?x?xf32>
        %17 = flow.dispatch.tensor.load %14, offsets = [0, 0], sizes = [%10, %11], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%10, %11} -> tensor<?x?xf32>
        %18 = linalg.matmul ins(%15, %16 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %18, %14, offsets = [0, 0], sizes = [%10, %11], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%10, %11}
        return
      }
    }
  }
}
//     CHECK: func private @vmvx.matmul.f32f32f32(
//     CHECK: func @matmul_f32()
//     CHECK:   func.call @vmvx.matmul.f32f32f32(
