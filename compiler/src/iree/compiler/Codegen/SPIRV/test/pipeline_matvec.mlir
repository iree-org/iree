// RUN: iree-opt --split-input-file --iree-gpu-test-target=pascal@vulkan \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-spirv-configuration-pipeline), iree-codegen-linalg-to-spirv-pipeline)))' \
// RUN:   %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @i4_dequant_unit_matmul_f16 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {
    iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
      compute = fp32|fp16|int32, storage = b32|b16, subgroup = shuffle|arithmetic, dot = none, mma = [],
      subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [65535, 65535, 65535]>>
    }>) {
    hal.executable.export @i4_dequant_unit_matmul_f16 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @i4_dequant_unit_matmul_f16() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x86x128xf16>>
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x1x4096xf16>>
        %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4096, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>> -> tensor<4096x86x128xi4>
        %6 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4096, 86, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>> -> tensor<4096x86x1xf16>
        %7 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [4096, 86, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>> -> tensor<4096x86x1xf16>
        %8 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [1, 1, 86, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x86x128xf16>> -> tensor<1x1x86x128xf16>
        %9 = tensor.empty() : tensor<1x1x4096xf16>
        %10 = tensor.empty() : tensor<4096x86x128xf16>
        %11 = linalg.fill ins(%cst : f16) outs(%9 : tensor<1x1x4096xf16>) -> tensor<1x1x4096xf16>
        %12 = linalg.generic {
            indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%5, %6, %7 : tensor<4096x86x128xi4>, tensor<4096x86x1xf16>, tensor<4096x86x1xf16>) outs(%10 : tensor<4096x86x128xf16>) {
        ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
          %14 = arith.extui %in : i4 to i32
          %15 = arith.uitofp %14 : i32 to f16
          %16 = arith.subf %15, %in_1 : f16
          %17 = arith.mulf %16, %in_0 : f16
          linalg.yield %17 : f16
        } -> tensor<4096x86x128xf16>
        %13 = linalg.generic {
            indexing_maps = [
                affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
                affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
                affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
        ins(%8, %12 : tensor<1x1x86x128xf16>, tensor<4096x86x128xf16>) outs(%11 : tensor<1x1x4096xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %14 = arith.mulf %in, %in_0 : f16
          %15 = arith.addf %14, %out : f16
          linalg.yield %15 : f16
        } -> tensor<1x1x4096xf16>
        flow.dispatch.tensor.store %13, %4, offsets = [0, 0, 0], sizes = [1, 1, 4096], strides = [1, 1, 1] : tensor<1x1x4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<1x1x4096xf16>>
        return
      }
    }
  }
}

//   CHECK-LABEL: spirv.func @i4_dequant_unit_matmul_f16()

//     CHECK-DAG: %[[CSTVEC4XI32_255:.+]] = spirv.Constant dense<255> : vector<4xi32>
//     CHECK-DAG: %[[CSTVEC4XI32_0:.+]] = spirv.Constant dense<0> : vector<4xi32>
//     CHECK-DAG: %[[CSTVEC2XI32_4:.+]] = spirv.Constant dense<4> : vector<2xi32>
//     CHECK-DAG: %[[CSTVEC2XI32_15:.+]] = spirv.Constant dense<15> : vector<2xi32>

//         CHECK: spirv.mlir.loop

// Load the quantized weight and get 8xi4 out of it.
//         CHECK:   %[[LOAD:.+]] = spirv.Load "StorageBuffer" %{{.+}} : vector<4xi32>
//         CHECK:   %[[SHUF01:.+]] = spirv.VectorShuffle [0 : i32, 1 : i32] %[[LOAD]], %[[LOAD]] : vector<4xi32>, vector<4xi32> -> vector<2xi32>
//         CHECK:   %[[LOW4:.+]] = spirv.BitwiseAnd %[[SHUF01]], %[[CSTVEC2XI32_15]] : vector<2xi32>
//         CHECK:   %[[HIGH4:.+]] = spirv.ShiftRightLogical %[[SHUF01]], %[[CSTVEC2XI32_4]] : vector<2xi32>, vector<2xi32>
//         CHECK:   %[[LOW4HIGH4:.+]] = spirv.VectorShuffle [0 : i32, 2 : i32, 1 : i32, 3 : i32] %[[LOW4]], %[[HIGH4]] : vector<2xi32>, {{.*}} -> vector<4xi32>
//         CHECK:   %[[LOW4HIGH4_ZEROUPPER:.+]] = spirv.BitwiseAnd %[[LOW4HIGH4]], %[[CSTVEC4XI32_255]] : vector<4xi32>

//         CHECK:   %[[SHUF23:.+]] = spirv.VectorShuffle [2 : i32, 3 : i32] %[[LOAD:.+]], %[[LOAD:.+]] : vector<4xi32>, vector<4xi32> -> vector<2xi32>

// CHECK-COUNT-2:   spirv.ConvertUToF %{{.+}} : vector<4xi32> to vector<4xf16>
// CHECK-COUNT-2:   spirv.FSub %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-4:   spirv.FMul %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:   spirv.FAdd %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:   spirv.Bitcast %{{.+}} : vector<4xf16> to vector<2xi32>
// CHECK-COUNT-2:   spirv.VectorShuffle {{.+}} : vector<4xi32>, vector<2xi32> -> vector<4xi32>

//         CHECK:   spirv.mlir.merge

//         CHECK: %[[LD:.+]] = spirv.Load "Function" {{.*}} : vector<4xi32>
//         CHECK: %[[VS0:.+]] = spirv.VectorShuffle [0 : i32, 1 : i32] %[[LD]]
//         CHECK: spirv.Bitcast %[[VS0]] : vector<2xi32> to vector<4xf16>
//         CHECK: %[[VS1:.+]] = spirv.VectorShuffle [2 : i32, 3 : i32] %[[LD]]
//         CHECK: spirv.Bitcast %[[VS1]] : vector<2xi32> to vector<4xf16>

//         CHECK: spirv.GroupNonUniformFAdd "Subgroup" "Reduce" {{.*}} : f16

//         CHECK: spirv.mlir.selection

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 5, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @i4_dequant_matvec_f16_subgroup_64 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {
    iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
      compute = fp32|fp16|int32, storage = b32|b16, subgroup = shuffle|arithmetic, dot = none, mma = [],
      subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [65535, 65535, 65535]>>
  }>) {
    hal.executable.export @i4_dequant_matvec_f16_subgroup_64 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @i4_dequant_matvec_f16_subgroup_64() {
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
        %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
        %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
        %5 = arith.index_castui %0 : i32 to index
        %6 = arith.index_castui %1 : i32 to index
        %7 = arith.index_castui %2 : i32 to index
        %8 = arith.index_castui %3 : i32 to index
        %9 = arith.index_castui %4 : i32 to index
        %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>>
        %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86xf16>>
        %12 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%7) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86xf16>>
        %13 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%8) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<86x128xf16>>
        %14 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%9) : !flow.dispatch.tensor<writeonly:tensor<4096xf16>>
        %15 = flow.dispatch.tensor.load %10, offsets = [0, 0, 0], sizes = [4096, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>> -> tensor<4096x86x128xi4>
        %16 = flow.dispatch.tensor.load %11, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf16>> -> tensor<4096x86xf16>
        %17 = flow.dispatch.tensor.load %12, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf16>> -> tensor<4096x86xf16>
        %18 = flow.dispatch.tensor.load %13, offsets = [0, 0], sizes = [86, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<86x128xf16>> -> tensor<86x128xf16>
        %19 = tensor.empty() : tensor<4096xf16>
        %20 = tensor.empty() : tensor<4096x86x128xf16>
        %21 = linalg.fill ins(%cst : f16) outs(%19 : tensor<4096xf16>) -> tensor<4096xf16>
        %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %16, %17 : tensor<4096x86x128xi4>, tensor<4096x86xf16>, tensor<4096x86xf16>) outs(%20 : tensor<4096x86x128xf16>) {
        ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
          %24 = arith.extui %in : i4 to i32
          %25 = arith.uitofp %24 : i32 to f16
          %26 = arith.subf %25, %in_1 : f16
          %27 = arith.mulf %26, %in_0 : f16
          linalg.yield %27 : f16
        } -> tensor<4096x86x128xf16>
        %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%18, %22 : tensor<86x128xf16>, tensor<4096x86x128xf16>) outs(%21 : tensor<4096xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %24 = arith.mulf %in, %in_0 : f16
          %25 = arith.addf %24, %out : f16
          linalg.yield %25 : f16
        } -> tensor<4096xf16>
        flow.dispatch.tensor.store %23, %14, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<4096xf16>>
        return

      }
    }
  }
}

//   CHECK-LABEL: spirv.func @i4_dequant_matvec_f16_subgroup_64()

//     CHECK-DAG: %[[C5504:.+]] = spirv.Constant 5504 : i32
//     CHECK-DAG: %[[C64:.+]] = spirv.Constant 64 : i32
//     CHECK-DAG: %[[C4:.+]] = spirv.Constant 4 : i32
//     CHECK-DAG: %[[C2:.+]] = spirv.Constant 2 : i32
//     CHECK-DAG: %[[C0:.+]] = spirv.Constant 0 : i32
//     CHECK-DAG: %[[CSTVEC4XF16_1:.+]] = spirv.Constant dense<1.000000e+00> : vector<4xf16>
//     CHECK-DAG: %[[CSTVEC4XI32_255:.+]] = spirv.Constant dense<255> : vector<4xi32>
//     CHECK-DAG: %[[CSTVEC2XI32_4:.+]] = spirv.Constant dense<4> : vector<2xi32>
//     CHECK-DAG: %[[CSTVEC2XI32_15:.+]] = spirv.Constant dense<15> : vector<2xi32>

//         CHECK: %[[WIDX:.+]] = spirv.CompositeExtract %{{.*}}[0 : i32] : vector<3xi32>
//         CHECK: %[[PCPTR:.+]] = spirv.AccessChain %{{.*}}[{{.*}}, %[[C0]]] : !spirv.ptr<!spirv.struct<(!spirv.array<5 x i32, stride=4> [0])>, PushConstant>, i32, i32
//         CHECK: %[[STREAMBINDING:.+]] = spirv.Load "PushConstant" %[[PCPTR]] : i32

//         CHECK: %[[RADDR:.+]] = spirv.mlir.addressof @{{.*}} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>

//         CHECK: spirv.mlir.loop

// Load the quantized weight and get 4xi4 out of it. Ensure that the offset
// calculation avoids excessive scaling down in computing the element offset.
//         CHECK:   spirv.IMul %{{.*}}, %[[C64]] : i32
//         CHECK:   spirv.IAdd %{{.*}}, %[[STREAMBINDING]] : i32
//         CHECK:   spirv.IMul %{{.*}}, %[[C5504]] : i32
//         CHECK:   spirv.IAdd %{{.*}}, %{{.*}} : i32
//         CHECK:   spirv.IMul %[[WIDX]], %[[C2]] : i32
//         CHECK:   spirv.IAdd %{{.*}}, %{{.*}} : i32
//         CHECK:   %[[OFFSET:.+]] = spirv.SDiv %{{.*}}, %[[C4]] : i32
//         CHECK:   %[[ACCESS:.+]] = spirv.AccessChain %[[RADDR]][{{.*}}, %[[OFFSET]]] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32
//         CHECK:   spirv.Load "StorageBuffer" %[[ACCESS]] : i32

//         CHECK:   spirv.ShiftRightLogical %{{.*}}, %[[CSTVEC2XI32_4]] : vector<2xi32>, vector<2xi32>
//         CHECK:   spirv.VectorShuffle [0 : i32, 2 : i32, 1 : i32, 3 : i32] %{{.*}} : vector<2xi32>, vector<2xi32> -> vector<4xi32>
//         CHECK:   spirv.BitwiseAnd %{{.*}}, %[[CSTVEC4XI32_255]] : vector<4xi32>

//         CHECK:   spirv.ConvertUToF %{{.+}} : vector<4xi32> to vector<4xf16>
//         CHECK:   spirv.FSub %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:   spirv.FMul %{{.+}}, %{{.+}} : vector<4xf16>
//         CHECK:   spirv.FAdd %{{.+}}, %{{.+}} : vector<4xf16>

//         CHECK:   spirv.mlir.merge

//         CHECK: %[[LD:.+]] = spirv.Load "Function" {{.*}} : vector<4xf16>
//         CHECK: %[[RES:.+]] = spirv.Dot %[[LD]], %[[CSTVEC4XF16_1]] : vector<4xf16> -> f16

//         CHECK: spirv.GroupNonUniformFAdd "Subgroup" "Reduce" %[[RES]] : f16

//         CHECK: spirv.mlir.selection
