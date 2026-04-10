// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-configure-target-executable-variants{target=rocm}, builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target, iree-codegen-gpu-check-resource-usage)))))" \
// RUN:   %s | FileCheck %s

// Regression test for dynamic batch_matmul with WMMAR3 on gfx1100.
//
// WMMAR3 (RDNA3) has accumulator layout outer={8,1} which requires an
// expand_shape on the output. With dynamic shapes, tensor.dim users on
// the forall result previously blocked ExpandDestinationForallOp from
// folding the expand into the output buffer. This caused a separate shared
// memory allocation for the output accumulator, exceeding the 65536-byte
// shared memory limit (LHS 16KB + RHS 32KB + output 32KB = 80KB > 65KB).

#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
hal.executable private @batch_matmul_dynamic_wmmar3 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @batch_matmul_24xDxDx128_f16_f32 ordinal(0)
      layout(#pipeline_layout)
      count(%dev: !hal.device, %arg1: index, %arg2: index, %arg3: index)
        -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @batch_matmul_24xDxDx128_f16_f32() {
        %c32_i64 = arith.constant 32 : i64
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
        %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
        %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
        %5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : i32
        %6 = arith.extui %0 : i32 to i64
        %7 = arith.extui %1 : i32 to i64
        %8 = arith.shli %7, %c32_i64 : i64
        %9 = arith.ori %6, %8 : i64
        %10 = arith.index_castui %9 : i64 to index
        %11 = arith.extui %2 : i32 to i64
        %12 = arith.extui %3 : i32 to i64
        %13 = arith.shli %12, %c32_i64 : i64
        %14 = arith.ori %11, %13 : i64
        %15 = arith.index_castui %14 : i64 to index
        %16 = arith.extui %4 : i32 to i64
        %17 = arith.extui %5 : i32 to i64
        %18 = arith.shli %17, %c32_i64 : i64
        %19 = arith.ori %16, %18 : i64
        %20 = arith.index_castui %19 : i64 to index
        %21:3 = util.assume.int
            %10<umin = 0, umax = 9007199254740991>,
            %15<umin = 0, umax = 9007199254740991>,
            %20<udiv = 128>
          : index, index, index
        %22 = iree_tensor_ext.dispatch.workload.ordinal %21#0, 0 : index
        %23 = iree_tensor_ext.dispatch.workload.ordinal %21#1, 1 : index
        %24 = iree_tensor_ext.dispatch.workload.ordinal %21#2, 2 : index
        %25 = hal.interface.binding.subspan layout(#pipeline_layout)
          binding(0) alignment(64) offset(%c0)
          flags("ReadOnly|Indirect")
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x?x128xf16>>{%22}
        %26 = hal.interface.binding.subspan layout(#pipeline_layout)
          binding(1) alignment(64) offset(%c0)
          flags("ReadOnly|Indirect")
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x?x128xf16>>{%23}
        %27 = hal.interface.binding.subspan layout(#pipeline_layout)
          binding(2) alignment(64) offset(%c0)
          flags(Indirect)
          : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x?x?xf32>>{%24, %24}
        %28 = iree_tensor_ext.dispatch.tensor.load %25,
          offsets = [0, 0, 0], sizes = [24, %22, 128], strides = [1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x?x128xf16>>{%22}
            -> tensor<24x?x128xf16>
        %29 = iree_tensor_ext.dispatch.tensor.load %26,
          offsets = [0, 0, 0], sizes = [24, %23, 128], strides = [1, 1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x?x128xf16>>{%23}
            -> tensor<24x?x128xf16>
        %30 = tensor.empty(%24, %24) : tensor<24x?x?xf32>
        %31 = linalg.fill ins(%cst : f32) outs(%30 : tensor<24x?x?xf32>)
          -> tensor<24x?x?xf32>
        %32 = linalg.batch_matmul
          indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                           affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
          ins(%28, %29 : tensor<24x?x128xf16>, tensor<24x?x128xf16>)
          outs(%31 : tensor<24x?x?xf32>) -> tensor<24x?x?xf32>
        iree_tensor_ext.dispatch.tensor.store %32, %27,
          offsets = [0, 0, 0], sizes = [24, %24, %24], strides = [1, 1, 1]
          : tensor<24x?x?xf32>
            -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x?x?xf32>>{%24, %24}
        return
      }
    }
  }
}

// Verify the full codegen pipeline completes (no shared memory overflow)
// and produces WMMA instructions.
// CHECK-LABEL: hal.executable private @batch_matmul_dynamic_wmmar3
// CHECK:         func.func @batch_matmul_24xDxDx128_f16_f32
// CHECK:           amdgpu.wmma
