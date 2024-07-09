// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-reorder-workgroups{strategy=swizzle logSwTile=3})))))" \
// RUN:   %s | FileCheck --check-prefix=SWIZZLE %s

// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-reorder-workgroups{strategy=transpose})))))" \
// RUN:   %s | FileCheck --check-prefix=TRANSPOSE %s

// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-reorder-workgroups{strategy=chipletgroup logCgTile=3})))))" \
// RUN:   %s | FileCheck --check-prefix=CHIPLETGROUP %s

// Make sure we use static workgroup counts instead of introducting
// `hal.interface.workgroup.count` ops. These are currently not supported on ROCm.

// SWIZZLE-LABEL: hal.executable private @main_dispatch_0 {
// SWIZZLE-LABEL: func.func @main_dispatch_0_matmul_transpose_b_32000x32000x4096_f16
// SWIZZLE-DAG:               %[[WG_X:.+]] = hal.interface.workgroup.id[0] : index
// SWIZZLE-DAG:               %[[WG_Y:.+]] = hal.interface.workgroup.id[1] : index
// SWIZZLE-NOT:               hal.interface.workgroup.count
// SWIZZLE-DAG:               %[[SEL_X:.+]] = arith.select %{{.+}}, %[[WG_X]]
// SWIZZLE-DAG:               %[[SEL_Y:.+]] = arith.select %{{.+}}, %[[WG_Y]]
// SWIZZLE-DAG:               affine.apply #{{.+}}()[%[[SEL_X]]]
// SWIZZLE-DAG:               affine.apply #{{.+}}()[%[[SEL_Y]]]
// SWIZZLE:                   return

// CHIPLETGROUP-LABEL: hal.executable private @main_dispatch_0 {
// CHIPLETGROUP-LABEL: func.func @main_dispatch_0_matmul_transpose_b_32000x32000x4096_f16
// CHIPLETGROUP-DAG:               %[[WG_X:.+]] = hal.interface.workgroup.id[0] : index
// CHIPLETGROUP-DAG:               %[[WG_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHIPLETGROUP-NOT:               hal.interface.workgroup.count
// CHIPLETGROUP-DAG:               %[[C250:.+]] = arith.constant 250 : index
// CHIPLETGROUP-DAG:               %[[C500:.+]] = arith.constant 500 : index
// CHIPLETGROUP:                   %[[MUL:.+]] = arith.muli %[[WG_Y]], %[[C250]] : index
// CHIPLETGROUP:                   %[[ADD:.+]] = arith.addi %[[MUL]], %[[WG_X]] : index
// CHIPLETGROUP:                   %[[CMP:.+]] = arith.cmpi ugt, %[[ADD]], %{{.+}} : index
// CHIPLETGROUP:                   %[[SELECT:.+]] = arith.select %[[CMP]], %[[ADD]], %{{.+}} : index
// CHIPLETGROUP:                   %[[REM:.+]] = arith.remui %[[SELECT]], %{{.+}} : index
// CHIPLETGROUP:                   %[[ADDI:.+]] = arith.addi %{{.+}}, %[[REM]] : index
// CHIPLETGROUP:                   %[[REMI:.+]] = arith.remui %[[SELECT]], %{{.+}} : index
// CHIPLETGROUP:                   %[[DIV:.+]] = arith.divui %[[REMI]], %{{.+}} : index
// CHIPLETGROUP-DAG:               affine.apply #{{.+}}()[%[[ADDI]]]
// CHIPLETGROUP-DAG:               affine.apply #{{.+}}()[%[[DIV]]]
// CHIPLETGROUP:                   return

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

hal.executable private @main_dispatch_0 {
hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @main_dispatch_0_matmul_transpose_b_32000x32000x4096_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUMatmulSimt, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>, workgroup_size = [64 : index, 16 : index, 1 : index]} {
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
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32000x32000xf16>>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %2 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
      %3 = flow.dispatch.tensor.load %0, offsets = [%2, 0], sizes = [%c64, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<?x4096xf16>
      %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
      %5 = flow.dispatch.tensor.load %0, offsets = [%4, 0], sizes = [%c128, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<?x4096xf16>
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
      flow.dispatch.tensor.store %cast_1, %1, offsets = [%9, %10], sizes = [%c64, %c128], strides = [1, 1] : tensor<?x?xf16> -> !flow.dispatch.tensor<writeonly:tensor<32000x32000xf16>>
      return
    }
  }
}
}
