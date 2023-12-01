// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy)))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable @dynamic_batch_matvec {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100"}>) {
    hal.executable.export @dynamic_batch_matvec layout(#pipeline_layout)
    builtin.module {
      func.func @dynamic_batch_matvec() {
        %c32_i64 = arith.constant 32 : i64
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = hal.interface.constant.load[4] : i32
        %13 = arith.index_castui %0 : i32 to index
        %18 = arith.index_castui %1 : i32 to index
        %19 = arith.index_castui %2 : i32 to index
        %24 = arith.index_castui %3 : i32 to index
        %29 = arith.index_castui %4 : i32 to index
        %30 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%19) : !flow.dispatch.tensor<writeonly:tensor<32x1x128xf16>>
        %31 = flow.dispatch.workload.ordinal %24, 0 : index
        %32 = flow.dispatch.workload.ordinal %29, 1 : index
        %33 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%13) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x1x?xf16>>{%31}
        %34 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%18) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x?x128xf16>>{%32}
        %35 = flow.dispatch.tensor.load %33, offsets = [0, 0, 0], sizes = [32, 1, %31], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x1x?xf16>>{%31} -> tensor<32x1x?xf16>
        %36 = flow.dispatch.tensor.load %34, offsets = [0, 0, 0], sizes = [32, %32, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x?x128xf16>>{%32} -> tensor<32x?x128xf16>
        %37 = tensor.empty() : tensor<32x1x128xf16>
        %38 = linalg.fill ins(%cst : f16) outs(%37 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
        %39 = linalg.batch_matmul ins(%35, %36 : tensor<32x1x?xf16>, tensor<32x?x128xf16>) outs(%38 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
        flow.dispatch.tensor.store %39, %30, offsets = [0, 0, 0], sizes = [32, 1, 128], strides = [1, 1, 1] : tensor<32x1x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x1x128xf16>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 1], [0, 0, 0, 32]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction>
// CHECK-LABEL: hal.executable.export public @dynamic_batch_matvec
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [32 : index, 1 : index, 1 : index]
//       CHECK: func.func @dynamic_batch_matvec()
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]