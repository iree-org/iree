// RUN: iree-opt --iree-codegen-llvmcpu-configuration-pipeline --iree-codegen-llvmcpu-lowering-pipeline='include-llvm-lowering=false' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @masked_attention() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %scale = arith.constant 0.5 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x48x4xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x48x5xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3x48xi8>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x3x5xf32>>
  %q = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 3, 4], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3x4xf32>> -> tensor<2x3x4xf32>
  %k = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, 48, 4], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x48x4xf32>> -> tensor<2x48x4xf32>
  %v = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [2, 48, 5], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x48x5xf32>> -> tensor<2x48x5xf32>
  %mask = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [2, 3, 48], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3x48xi8>> -> tensor<2x3x48xi8>
  %empty = tensor.empty() : tensor<2x3x5xf32>
  %result = iree_linalg_ext.attention
      <indexing_maps = [affine_map<(b, m, k1, k2, n) -> (b, m, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, n)>,
                        affine_map<(b, m, k1, k2, n) -> ()>,
                        affine_map<(b, m, k1, k2, n) -> (b, m, k2)>,
                        affine_map<(b, m, k1, k2, n) -> (b, m, n)>]>
      ins(%q, %k, %v, %scale, %mask : tensor<2x3x4xf32>, tensor<2x48x4xf32>, tensor<2x48x5xf32>, f32, tensor<2x3x48xi8>)
      outs(%empty : tensor<2x3x5xf32>) {
    ^bb0(%s: f32):
      iree_linalg_ext.yield %s : f32
  } -> tensor<2x3x5xf32>
  iree_tensor_ext.dispatch.tensor.store %result, %4, offsets = [0, 0, 0], sizes = [2, 3, 5], strides = [1, 1, 1] : tensor<2x3x5xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x3x5xf32>>
  return
}

// The mask of each four-key tile is reduced first; K and V are read only when
// some key is visible.
// CHECK-LABEL: func.func @masked_attention
//       CHECK:   scf.for
//   CHECK-NOT:     memref<2x48x4xf32
//   CHECK-NOT:     memref<2x48x5xf32
//       CHECK:     vector.load {{.+}} : memref<2x3x48xi8, {{.+}}>, vector<4xi8>
//       CHECK:     vector.reduction <or>
//       CHECK:     scf.if
//       CHECK:       vector.load {{.+}} : memref<2x48x4xf32, {{.+}}>
//       CHECK:       math.exp2
//       CHECK:       vector.load {{.+}} : memref<2x48x5xf32, {{.+}}>
//       CHECK:       memref.store
//  CHECK-NEXT:     }
//  CHECK-NEXT:   }
