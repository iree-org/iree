// RUN: iree-opt -pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy)))' -split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_static  {
  hal.executable.variant @vmvx_bytecode_fb target(<"vmvx", "vmvx-bytecode-fb">) {
    hal.executable.export public @matmul_static layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_static() {
        %cst = arith.constant 0.0 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x128xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [384, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [512, 128], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x128xf32>
        %init = tensor.empty() : tensor<384x128xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x128xf32>) -> tensor<384x128xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<384x512xf32>, tensor<512x128xf32>)
            outs(%fill : tensor<384x128xf32>) -> tensor<384x128xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [384, 128], strides = [1, 1]
            : tensor<384x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//      CHECK: hal.executable.export public @matmul_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @copy_op_dynamic {
  hal.executable.variant @vmvx_bytecode_fb target(<"vmvx", "vmvx-bytecode-fb">) {
    hal.executable.export @copy_op_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @copy_op_dynamic() {
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %d2 = hal.interface.constant.load[2] : index
        %d3 = hal.interface.constant.load[3] : index
        %o0 = hal.interface.constant.load[4] : index
        %o1 = hal.interface.constant.load[5] : index
        %source = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xi32>{%d0, %d1}
        %dest = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xi32>{%d2, %d3}
        %dest_view = memref.subview %dest[%o0, %o1] [%d0, %d1] [1, 1] : memref<?x?xi32> to memref<?x?xi32, strided<[?, ?], offset : ?>>
        linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)> , affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%source : memref<?x?xi32>) outs(%dest_view : memref<?x?xi32, strided<[?, ?], offset : ?>>) {
          ^bb0(%arg0 : i32, %arg1 : i32):
            linalg.yield %arg0 : i32
          }
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//      CHECK: hal.executable.export public @copy_op_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_1d_fft_stage2  {
  hal.executable.variant @vmvx_bytecode_fb target(<"vmvx", "vmvx-bytecode-fb">) {
    hal.executable.export @static_1d_fft_stage2 layout(#pipeline_layout)
    builtin.module {
      func.func @static_1d_fft_stage2() {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
        %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
        %4:2 = iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
        flow.dispatch.tensor.store %4#0, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        flow.dispatch.tensor.store %4#1, %1, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//       CHECK: hal.executable.export public @static_1d_fft_stage2
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: func.func @static_1d_fft_stage2()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable @fusion_quant_matmul_generic {
  hal.executable.variant @vmvx_bytecode_fb target(<"vmvx", "vmvx-bytecode-fb">) {
    hal.executable.export @fusion_quant_matmul_generic layout(#pipeline_layout)
    builtin.module {
      func.func @fusion_quant_matmul_generic() {
        %c0_i32 = arith.constant 0 : i32
        %c-128_i32 = arith.constant -128 : i32
        %c1101627623_i32 = arith.constant 1101627623 : i32
        %c36_i8 = arith.constant 36 : i8
        %c127_i32 = arith.constant 127 : i32
        %c107520 = arith.constant 107520 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3360x32xi8>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32xi32>>
        %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c107520) : !flow.dispatch.tensor<readonly:tensor<32xi32>>
        %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x3360xi8>>{%1}
        %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x32xi8>>{%1}
        %7 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%1, 3360], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x3360xi8>>{%1} -> tensor<?x3360xi8>
        %8 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [3360, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<3360x32xi8>> -> tensor<3360x32xi8>
        %9 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readonly:tensor<32xi32>> -> tensor<32xi32>
        %10 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readonly:tensor<32xi32>> -> tensor<32xi32>
        %11 = tensor.empty(%1) : tensor<?x32xi8>
        %12 = tensor.empty(%1) : tensor<?x32xi32>
        %13 = linalg.fill ins(%c0_i32 : i32) outs(%12 : tensor<?x32xi32>) -> tensor<?x32xi32>
        %14 = linalg.matmul ins(%7, %8 : tensor<?x3360xi8>, tensor<3360x32xi8>) outs(%13 : tensor<?x32xi32>) -> tensor<?x32xi32>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %14, %10 : tensor<32xi32>, tensor<?x32xi32>, tensor<32xi32>) outs(%11 : tensor<?x32xi8>) {
        ^bb0(%in: i32, %in_0: i32, %in_1: i32, %out: i8):
          %16 = arith.muli %in_1, %c-128_i32 : i32
          %17 = arith.subi %in_0, %16 : i32
          %18 = arith.addi %in, %17 : i32
          %19 = tosa.apply_scale %18, %c1101627623_i32, %c36_i8 {double_round = true} : (i32, i32, i8) -> i32
          %20 = arith.addi %19, %c-128_i32 : i32
          %21 = arith.cmpi slt, %20, %c-128_i32 : i32
          %22 = arith.select %21, %c-128_i32, %20 : i32
          %23 = arith.cmpi sgt, %20, %c127_i32 : i32
          %24 = arith.select %23, %c127_i32, %22 : i32
          %25 = arith.trunci %24 : i32 to i8
          linalg.yield %25 : i8
        } -> tensor<?x32xi8>
        flow.dispatch.tensor.store %15, %6, offsets = [0, 0], sizes = [%1, 32], strides = [1, 1] : tensor<?x32xi8> -> !flow.dispatch.tensor<writeonly:tensor<?x32xi8>>{%1}
        return
      }
    }
  }
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 16, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//       CHECK: hal.executable.export public @fusion_quant_matmul_generic
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: func.func @fusion_quant_matmul_generic()
//       CHECK:   linalg.matmul
//  CHECK-SAME:       lowering_config = #[[CONFIG]]


// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @unpack_outer_dynamic  {
  hal.executable.variant @vmvx_bytecode_fb target(<"vmvx", "vmvx-bytecode-fb">) {
  hal.executable.export public @unpack_outer_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @unpack_outer_dynamic() {
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
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5}
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c131072) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        %10 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [%4, %5, 32, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5} -> tensor<?x?x32x16xi32>
        %11 = tensor.empty(%6, %7) : tensor<?x?xi32>
        %12 = tensor.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %11 : tensor<?x?x32x16xi32> -> tensor<?x?xi32>
        flow.dispatch.tensor.store %12, %9, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [32, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//      CHECK: hal.executable.export public @unpack_outer_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.unpack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

hal.executable private @elem_pack_ukernels  {
  hal.executable.variant public @vmvx_bytecode_fb target(<"vmvx", "vmvx-bytecode-fb", {ukernels = true}>) {
    hal.executable.export public @elem_pack_ukernels ordinal(0) layout(#hal.pipeline.layout<push_constants = 8, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @elem_pack_ukernels() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x2048xf32>>
        %1:2 = iree_codegen.query_tile_sizes tensor<1024x2048xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  LHS, element_types = [f32, f32, f32], original_type = tensor<1024x2048xf32>>> -> index, index
        %2 = affine.apply affine_map<()[s0] -> (1024 ceildiv s0)>()[%1#0]
        %3 = affine.apply affine_map<()[s0] -> (2048 ceildiv s0)>()[%1#1]
        %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%2, %3, %1#0, %1#1}
        %5 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x2048xf32>> -> tensor<1024x2048xf32>
        %6 = tensor.empty() : tensor<1024x2048xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1024x2048xf32>) outs(%6 : tensor<1024x2048xf32>) {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.addf %in, %in : f32
          linalg.yield %15 : f32
        } -> tensor<1024x2048xf32>
        %8:2 = iree_codegen.query_tile_sizes tensor<?x?xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  LHS, element_types = [f32, f32, f32], original_type = tensor<1024x2048xf32>>> -> index, index
        %9 = affine.apply affine_map<()[s0] -> (1024 ceildiv s0)>()[%8#0]
        %10 = affine.apply affine_map<()[s0] -> (2048 ceildiv s0)>()[%8#1]
        %11 = tensor.empty(%9, %10, %8#0, %8#1) : tensor<?x?x?x?xf32>
        %pack = tensor.pack %7 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [%8#0, %8#1] into %11 : tensor<1024x2048xf32> -> tensor<?x?x?x?xf32>
        %12:2 = iree_codegen.query_tile_sizes tensor<1024x2048xf32, #iree_linalg_ext.encoding<user =  MATMUL, role =  LHS, element_types = [f32, f32, f32], original_type = tensor<1024x2048xf32>>> -> index, index
        %13 = affine.apply affine_map<()[s0] -> (1024 ceildiv s0)>()[%12#0]
        %14 = affine.apply affine_map<()[s0] -> (2048 ceildiv s0)>()[%12#1]
        flow.dispatch.tensor.store %pack, %4, offsets = [0, 0, 0, 0], sizes = [%13, %14, %12#0, %12#1], strides = [1, 1, 1, 1] : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%13, %14, %12#0, %12#1}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<VMVXDefault>
//      CHECK: hal.executable.export public @elem_pack_ukernels
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]
