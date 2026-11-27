// RUN: iree-opt --iree-codegen-llvmcpu-configuration-pipeline --iree-codegen-llvmcpu-lowering-pipeline='include-llvm-lowering=false' --split-input-file %s | FileCheck %s

// Attention leaves the pack after it unfused; the pack then takes its own
// vector tiles over the buffered attention result instead of one vector for
// the whole workgroup.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "znver4", cpu_features = "+avx512f,+avx512vl,+avx512bw,+avx512dq,+avx2,+fma", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", native_vector_size = 64 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
func.func @attention_pack_consumer() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %scale = arith.constant 0.125 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x2x256x32xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x80x32xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<80x3x64xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x80xi8>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xi8>>
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(5) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x3x2x64x16x1xf32>>
  %q = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [3, 2, 256, 32], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x2x256x32xf32>> -> tensor<3x2x256x32xf32>
  %k = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 80, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x80x32xf32>> -> tensor<3x80x32xf32>
  %v = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [80, 3, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<80x3x64xf32>> -> tensor<80x3x64xf32>
  %mask = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [256, 80], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x80xi8>> -> tensor<256x80xi8>
  %rows = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0], sizes = [256], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xi8>> -> tensor<256xi8>
  %empty = tensor.empty() : tensor<3x2x256x64xf32>
  %attn = iree_linalg_ext.attention <indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d4)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d0, d3)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d5)>,
      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]>
      ins(%q, %k, %v, %scale, %mask : tensor<3x2x256x32xf32>, tensor<3x80x32xf32>, tensor<80x3x64xf32>, f32, tensor<256x80xi8>)
      outs(%empty : tensor<3x2x256x64xf32>) {
    ^bb0(%s: f32):
      iree_linalg_ext.yield %s : f32
  } -> tensor<3x2x256x64xf32>
  %t_empty = tensor.empty() : tensor<256x3x2x64xf32>
  %t = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%attn, %rows : tensor<3x2x256x64xf32>, tensor<256xi8>) outs(%t_empty : tensor<256x3x2x64xf32>) {
    ^bb0(%in: f32, %row: i8, %out: f32):
      %visible = arith.trunci %row : i8 to i1
      %r = arith.select %visible, %in, %zero : f32
      linalg.yield %r : f32
  } -> tensor<256x3x2x64xf32>
  %p_empty = tensor.empty() : tensor<16x3x2x64x16x1xf32>
  %pack = linalg.pack %t outer_dims_perm = [0, 1, 2, 3] inner_dims_pos = [0, 3] inner_tiles = [16, 1] into %p_empty : tensor<256x3x2x64xf32> -> tensor<16x3x2x64x16x1xf32>
  iree_tensor_ext.dispatch.tensor.store %pack, %5, offsets = [0, 0, 0, 0, 0, 0], sizes = [16, 3, 2, 64, 16, 1], strides = [1, 1, 1, 1, 1, 1] : tensor<16x3x2x64x16x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x3x2x64x16x1xf32>>
  return
}
// CHECK-LABEL: func.func @attention_pack_consumer
//       CHECK:   %[[RESULT:.+]] = memref.alloca() {{.*}} : memref<32x1x2x64xf32>
//       CHECK:   scf.for {{.+}} = %c0 to %c2 step %c1 {
//       CHECK:     scf.for {{.+}} = %c0 to %c32 step %c8 {
//       CHECK:       scf.for {{.+}} = %c0 to %c80 step %c8
//       CHECK:   scf.for {{.+}} = %c0 to %c2 step %c1 {
//  CHECK-NEXT:     vector.load %[[RESULT]]{{.+}} : memref<32x1x2x64xf32>, vector<64xf32>
