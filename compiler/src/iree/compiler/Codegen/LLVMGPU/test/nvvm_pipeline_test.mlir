// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_60 --iree-codegen-llvmgpu-configuration-pipeline --iree-codegen-llvmgpu-nvvm-lowering-pipeline %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_80 --iree-codegen-llvmgpu-configuration-pipeline --iree-codegen-llvmgpu-nvvm-lowering-pipeline %s | FileCheck %s --check-prefix=SM80

// Verify that a simple element wise op gets lowered successfully all the way to
// nvvm/llvm dialect.

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @add_dispatch_0() attributes {hal.executable.target = #executable_target_cuda} {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %3 = tensor.empty() : tensor<16xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets=[0], sizes=[16], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
  %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets=[0], sizes=[16], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %7 = arith.addf %arg0, %arg1 : f32
      linalg.yield %7 : f32
    } -> tensor<16xf32>
    iree_tensor_ext.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
    return
}

// CHECK-LABEL: llvm.func @add_dispatch_0
//       CHECK:   llvm.fadd

// -----

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dot_dispatch_0() attributes {hal.executable.target = #executable_target_cuda} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
  %10 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
  %15 = tensor.empty() : tensor<1024x1024xf32>
  %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %17 = linalg.matmul ins(%8, %10 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
      outs(%16 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  iree_tensor_ext.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
      : tensor<1024x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  return
}

//      CHECK-LABEL: llvm.func @dot_dispatch_0
//        CHECK-NOT:   llvm.store
//            CHECK:   llvm.br
//            CHECK:    llvm.load {{.*}} : !llvm.ptr<1> -> vector<32xf32>
//   CHECK-COUNT-32:    llvm.load {{.*}} : !llvm.ptr<1> -> vector<16xf32>
//   CHECK-COUNT-512:  llvm.call @__nv_fmaf({{.*}}) : (f32, f32, f32) -> f32
//            CHECK:    llvm.store {{.*}} : vector<16xf32>, !llvm.ptr<1>

// -----

// Check that a generic op representing a matmul is getting the same
// configuration as the matmul op.
#matmul_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}
#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dot_dispatch_0() attributes {hal.executable.target = #executable_target_cuda} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
  %10 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
  %15 = tensor.empty() : tensor<1024x1024xf32>
  %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %17 = linalg.generic #matmul_trait
      ins(%8, %10 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%16 : tensor<1024x1024xf32>)  {
    ^bb(%a: f32, %b: f32, %c: f32) :
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
    } -> (tensor<1024x1024xf32>)
  iree_tensor_ext.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
      : tensor<1024x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  return
}

//   CHECK-LABEL: llvm.func @dot_dispatch_0
//            CHECK:  llvm.br
//   CHECK-COUNT-512:  llvm.call @__nv_fmaf({{.*}}) : (f32, f32, f32) -> f32
//            CHECK:    llvm.store {{.*}} : vector<16xf32>, !llvm.ptr<1>

// -----

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @conv2d_dispatch_0() attributes {hal.executable.target = #executable_target_cuda} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x4x2xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x2x2x1xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2x3x1xf32>>
  %11 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0 ,0, 0], sizes = [1, 4, 4, 2], strides = [1, 1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x4x2xf32>> -> tensor<1x4x4x2xf32>
  %13 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 2, 2, 1], strides = [1, 1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x2x2x1xf32>> -> tensor<3x2x2x1xf32>
  %20 = tensor.empty() : tensor<1x2x3x1xf32>
  %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x2x3x1xf32>) -> tensor<1x2x3x1xf32>
  %22 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%11, %13 : tensor<1x4x4x2xf32>, tensor<3x2x2x1xf32>) outs(%21 : tensor<1x2x3x1xf32>) -> tensor<1x2x3x1xf32>
  iree_tensor_ext.dispatch.tensor.store %22, %2, offsets = [0, 0, 0, 0], sizes = [1, 2, 3, 1], strides = [1, 1, 1, 1]
      : tensor<1x2x3x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2x3x1xf32>>
  return
}

//   CHECK-LABEL: llvm.func @conv2d_dispatch_0
// CHECK-COUNT-3:   llvm.load %{{.*}} : !llvm.ptr<1> -> f32
//         CHECK:   llvm.fmul %{{.*}}, %{{.*}}  : f32
//         CHECK:   llvm.fadd %{{.*}}, %{{.*}}  : f32
//         CHECK:   llvm.store {{.*}} : f32, !llvm.ptr<1>

// -----

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @add_dispatch_0() attributes {hal.executable.target = #executable_target_cuda} {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %3 = tensor.empty() : tensor<16xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets=[0], sizes=[16], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
  %5 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %7 = arith.addf %arg0, %arg1 : f32
      linalg.yield %7 : f32
  } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// CHECK: llvm.mlir.global private constant @{{.*}}(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<16xf32>)
// CHECK-LABEL: llvm.func @add_dispatch_0
//       CHECK:   llvm.fadd

// -----

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @reduction() attributes {hal.executable.target = #executable_target_cuda} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<14x14x96xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96xf32>>
  %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [14, 14, 96], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<14x14x96xf32>> -> tensor<14x14x96xf32>
  %8 = tensor.empty() : tensor<96xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<96xf32>) -> tensor<96xf32>
  %10 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0)>],
        iterator_types = ["parallel", "reduction", "reduction"]}
        ins(%5 : tensor<14x14x96xf32>) outs(%9 : tensor<96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %11 = arith.addf %arg1, %arg2 : f32
      linalg.yield %11 : f32
    } -> tensor<96xf32>
  iree_tensor_ext.dispatch.tensor.store %10, %1, offsets = [0], sizes = [96], strides = [1]
      : tensor<96xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96xf32>>
  return
}

// CHECK-LABEL: llvm.func @reduction
//       CHECK:     "llvm.intr.vector.reduce.fadd"({{.*}}) {{.*}} : (f32, vector<4xf32>) -> f32

// -----

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @vector_add_dispatch() attributes {hal.executable.target = #executable_target_cuda} {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
  %6 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [16384], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
  %8 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [16384], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
  %10 = tensor.empty() : tensor<16384xf32>
  %11 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%6, %8 : tensor<16384xf32>, tensor<16384xf32>) outs(%10 : tensor<16384xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %12 = arith.addf %arg1, %arg2 : f32
      linalg.yield %12 : f32
    } -> tensor<16384xf32>
  iree_tensor_ext.dispatch.tensor.store %11, %2, offsets = [0], sizes = [16384], strides = [1]
      : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
  return
}

//   CHECK-LABEL: llvm.func @vector_add_dispatch
//         CHECK:   llvm.fadd %{{.*}}, %{{.*}}  : vector<4xf32
//         CHECK:   llvm.store %{{.*}} : vector<4xf32>, !llvm.ptr<1>

// -----

#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> (d0)>
#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @vector_reduction_dispatch() attributes {hal.executable.target = #executable_target_cuda} {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x16384xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
  %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 16384], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x16384xf32>> -> tensor<512x16384xf32>
  %8 = tensor.empty() : tensor<16384xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<16384xf32>) -> tensor<16384xf32>
  %10 = linalg.generic {
      indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]}
      ins(%5 : tensor<512x16384xf32>) outs(%9 : tensor<16384xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %11 = arith.addf %arg1, %arg2 : f32
      linalg.yield %11 : f32
    } -> tensor<16384xf32>
  iree_tensor_ext.dispatch.tensor.store %10, %1, offsets = [0], sizes = [16384], strides = [1]
      : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
  return
}

//   CHECK-LABEL: llvm.func @vector_reduction_dispatch
// CHECK-COUNT-4:     "llvm.intr.vector.reduce.fadd"({{.*}}) {{.*}} : (f32, vector<4xf32>) -> f32
//         CHECK:     llvm.store %{{.*}} : vector<4xf32>, !llvm.ptr<1>

// -----

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @pooling_dynamic() attributes {hal.executable.target = #executable_target_cuda} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %cast = arith.index_cast %0 : i32 to index
  %s = iree_tensor_ext.dispatch.workload.ordinal %cast, 0 : index
  %14 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%s) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x?x?xf32>>{%s, %s, %s}
  %15 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%s) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048x1x1xf32>>{%s}
  %16 = iree_tensor_ext.dispatch.tensor.load %14, offsets = [0, 0, 0, 0], sizes = [%s, 2048, %s, %s], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048x?x?xf32>>{%s, %s, %s} -> tensor<?x2048x?x?xf32>
  %19 = tensor.empty(%s) : tensor<?x2048x1x1xf32>
  %38 = tensor.empty(%s, %s) : tensor<?x?xf32>
  %39 = linalg.fill ins(%cst : f32) outs(%19 : tensor<?x2048x1x1xf32>) -> tensor<?x2048x1x1xf32>
  %40 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%16, %38 : tensor<?x2048x?x?xf32>, tensor<?x?xf32>) outs(%39 : tensor<?x2048x1x1xf32>) -> tensor<?x2048x1x1xf32>
  iree_tensor_ext.dispatch.tensor.store %40, %15, offsets = [0, 0, 0, 0], sizes = [%s, 2048, 1, 1], strides = [1, 1, 1, 1] : tensor<?x2048x1x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048x1x1xf32>>{%s}
  return
}

// Just check that compilation succeed.
//     SM80-LABEL: llvm.func @pooling_dynamic

// -----

#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @vector_distribute_dispatch() attributes {hal.executable.target = #executable_target_cuda} {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512xf32>>
  %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
  %8 = tensor.empty() : tensor<512xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<512xf32>) -> tensor<512xf32>
  %10 = linalg.generic {
      indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]}
      ins(%5 : tensor<512x1024xf32>) outs(%9 : tensor<512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %11 = arith.addf %arg1, %arg2 : f32
      linalg.yield %11 : f32
    } -> tensor<512xf32>
  iree_tensor_ext.dispatch.tensor.store %10, %1, offsets = [0], sizes = [512], strides = [1]
      : tensor<512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512xf32>>
  return
}

// Check that we generate a vector distribute code sequence.
//   CHECK-LABEL: llvm.func @vector_distribute_dispatch
// CHECK-COUNT-5:     nvvm.shfl.sync bfly
//         CHECK:     llvm.store %{{.*}}, %{{.*}} : vector<1xf32>, !llvm.ptr<3>
//         CHECK:     nvvm.barrier
//         CHECK:     llvm.load {{.*}} : !llvm.ptr<3> -> vector<1xf32>
// CHECK-COUNT-2:     nvvm.shfl.sync bfly

// -----

#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @vector_distribution_broadcast_dispatch() attributes {hal.executable.target = #executable_target_cuda} {
  %cst_0 = arith.constant 3.840000e+02 : f32
  %cst = arith.constant 1.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x1024xf32>>
  %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
  %8 = tensor.empty() : tensor<512xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<512xf32>) -> tensor<512xf32>
  %10 = linalg.generic {
      indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]}
      ins(%5 : tensor<512x1024xf32>) outs(%9 : tensor<512xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %11 = arith.addf %arg1, %arg2 : f32
      linalg.yield %11 : f32
    } -> tensor<512xf32>
  %i = tensor.empty() : tensor<512x1024xf32>
  %11 = linalg.generic {
    indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%10 : tensor<512xf32>) outs(%i : tensor<512x1024xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %12 = arith.divf %arg0, %cst_0 : f32
        linalg.yield %12 : f32
      } -> tensor<512x1024xf32>
  iree_tensor_ext.dispatch.tensor.store %11, %1, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1]
      : tensor<512x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x1024xf32>>
  return
}

// Check that we generate a group reduce fused with broadcast + elementwise.
//   CHECK-LABEL: llvm.func @vector_distribution_broadcast_dispatch
// CHECK-COUNT-5:     nvvm.shfl.sync bfly
//         CHECK:     llvm.store %{{.*}}, %{{.*}} : vector<1xf32>, !llvm.ptr<3>
//         CHECK:     nvvm.barrier
//         CHECK:     llvm.load {{.*}} : !llvm.ptr<3> -> vector<1xf32>
// CHECK-COUNT-2:     nvvm.shfl.sync bfly
//         CHECK:     llvm.fdiv %{{.*}}, %{{.*}}
//         CHECK:     llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<1>

// -----

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @generalized_pool() attributes {hal.executable.target = #executable_target_cuda} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<14x14x480xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<14x14x480xf32>) -> tensor<14x14x480xf32>
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<29x29x480xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14x14x480xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [29, 29, 480], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<29x29x480xf32>> -> tensor<29x29x480xf32>
  %3 = tensor.empty() : tensor<3x3xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%2, %3 : tensor<29x29x480xf32>, tensor<3x3xf32>) outs(%fill : tensor<14x14x480xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %5 = arith.maximumf %arg2, %arg0 : f32
    linalg.yield %5 : f32
  } -> tensor<14x14x480xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [14, 14, 480], strides = [1, 1, 1] : tensor<14x14x480xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14x14x480xf32>>
  return
}

//   CHECK-LABEL: llvm.func @generalized_pool
//         CHECK:     llvm.load %{{.*}} : !llvm.ptr<1> -> f32
//         CHECK:     llvm.call @__nv_fmaxf
//         CHECK:     llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<1>

// -----

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @shared_mem_transpose() attributes {hal.executable.target = #executable_target_cuda} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x768xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<768x2048xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 768], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x768xf32>> -> tensor<2048x768xf32>
  %3 = tensor.empty() : tensor<768x2048xf32>
  %4 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<2048x768xf32>) outs(%3 : tensor<768x2048xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    linalg.yield %arg0 : f32
  } -> tensor<768x2048xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [768, 2048], strides = [1, 1] : tensor<768x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<768x2048xf32>>
  return
}

// Check that bufferization is emitting correct code for the temp shared
// memory alloc.
//   SM80-LABEL: llvm.func @shared_mem_transpose
//         SM80:     llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> vector<4xf32>
//         SM80:     llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<3>

// -----

// Verify that an f16 matmul on sm_80 lowers all the way to nvvm.mma.sync
// intrinsics (Tensor Core path). On sm_60, the same op falls back to scalar
// FMA since Tensor Cores are unavailable.

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_f16() attributes {hal.executable.target = #executable_target_cuda} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
  %5 = tensor.empty() : tensor<256x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  return
}

//    SM80-LABEL: llvm.func @matmul_f16
//     SM80-NOT:     nvgpu.mma.sync
// SM80-COUNT-64: nvvm.mma.sync{{.*}}shape = #nvvm.shape<m = 16, n = 8, k = 16>
//     SM80-NOT:     nvvm.mma.sync
//         SM80:     llvm.return
