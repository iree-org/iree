// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-reorder-workgroups{strategy=transpose})))))" \
// RUN:   %s | FileCheck --check-prefix=TRANSPOSE %s

// Make sure we use static workgroup counts instead of introducting
// `hal.interface.workgroup.count` ops. These are currently not supported on ROCm.

// TRANSPOSE-LABEL: hal.executable private @main_dispatch_0 {
// TRANSPOSE-LABEL: func.func @main_dispatch_0_matmul_transpose_b_32000x32000x4096_f16
// TRANSPOSE-DAG:               %[[WG_X:.+]] = hal.interface.workgroup.id[0] : index
// TRANSPOSE-DAG:               %[[WG_Y:.+]] = hal.interface.workgroup.id[1] : index
// TRANSPOSE-NOT:               hal.interface.workgroup.count
// TRANSPOSE-DAG:               %[[C250:.+]] = arith.constant 250 : index
// TRANSPOSE-DAG:               %[[C500:.+]] = arith.constant 500 : index
// TRANSPOSE:                   %[[MUL:.+]] = arith.muli %[[WG_Y]], %[[C250]] : index
// TRANSPOSE:                   %[[ADD:.+]] = arith.addi %[[MUL]], %[[WG_X]] : index
// TRANSPOSE:                   %[[DIV:.+]] = arith.divui %[[ADD]], %[[C500]] : index
// TRANSPOSE:                   %[[REM:.+]] = arith.remui %[[ADD]], %[[C500]] : index
// TRANSPOSE-DAG:               affine.apply #{{.+}}()[%[[DIV]]]
// TRANSPOSE-DAG:               affine.apply #{{.+}}()[%[[REM]]]
// TRANSPOSE:                   return

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @main_dispatch_0 {
hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @main_dispatch_0_matmul_transpose_b_32000x32000x4096_f16 ordinal(0) layout(#pipeline_layout) attributes {subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>, workgroup_size = [64 : index, 16 : index, 1 : index]} {
  ^bb0(%arg0: !hal.device):
    %c250 = arith.constant 250 : index
    %c500 = arith.constant 500 : index
    %c1 = arith.constant 1 : index
    hal.return %c250, %c500, %c1 : index, index, index
  }
  builtin.module {
    func.func @main_dispatch_0_matmul_transpose_b_32000x32000x4096_f16() {
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x32000xf16>>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %2 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%2, 0], sizes = [%c64, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<?x4096xf16>
      %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
      %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%4, 0], sizes = [%c128, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<?x4096xf16>
      %6 = tensor.empty() : tensor<64x128xf16>
      %7 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f16) outs(%6 : tensor<64x128xf16>) -> tensor<64x128xf16>
      %cast = tensor.cast %5 : tensor<?x4096xf16> to tensor<128x4096xf16>
      %cast_0 = tensor.cast %3 : tensor<?x4096xf16> to tensor<64x4096xf16>
      %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cast_0, %cast : tensor<64x4096xf16>, tensor<128x4096xf16>) outs(%7 : tensor<64x128xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {
      ^bb0(%in: f16, %in_2: f16, %out: f16):
        %11 = arith.mulf %in, %in_2 : f16
        %12 = arith.addf %out, %11 : f16
        linalg.yield %12 : f16
      } -> tensor<64x128xf16>
      %cast_1 = tensor.cast %8 : tensor<64x128xf16> to tensor<?x?xf16>
      %9 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
      %10 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
      iree_tensor_ext.dispatch.tensor.store %cast_1, %1, offsets = [%9, %10], sizes = [%c64, %c128], strides = [1, 1] : tensor<?x?xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x32000xf16>>
      return
    }
  }
}
}
