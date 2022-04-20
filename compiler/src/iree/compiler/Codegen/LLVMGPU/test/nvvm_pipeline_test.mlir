// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-codegen-linalg-to-nvvm-pipeline))' -iree-codegen-cuda-pipeline-depth=1 %s | FileCheck %s
// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-codegen-linalg-to-nvvm-pipeline))' -iree-codegen-cuda-pipeline-depth=4 %s | FileCheck %s -check-prefix=CHECKP

// Verify that a simple element wise op gets lowered succefully all the way to
// nvvm/llvm dialect.

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @simpleMath_ex_dispatch_0 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @add_dispatch_0 layout(#executable_layout)
  builtin.module {
    func.func @add_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:16xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:16xf32>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:16xf32>
      %3 = linalg.init_tensor [16] : tensor<16xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %5 = flow.dispatch.tensor.load %1, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16xf32>
        flow.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @simpleMath_ex_dispatch_0
//       CHECK:   hal.executable.variant public @cuda
//       CHECK:   llvm.fadd

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @dot_dispatch_0 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @dot_dispatch_0 layout(#executable_layout)
    builtin.module {
      func.func @dot_dispatch_0() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:1024x1024xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:1024x1024xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:1024x1024xf32>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:1024x1024xf32> -> tensor<1024x1024xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:1024x1024xf32> -> tensor<1024x1024xf32>
        %15 = linalg.init_tensor [1024, 1024] : tensor<1024x1024xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
        %17 = linalg.matmul ins(%8, %10 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
            outs(%16 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:1024x1024xf32>
        return
      }
    }
  }
}

//     CHECK-LABEL: hal.executable public @dot_dispatch_0
//           CHECK:   hal.executable.variant public @cuda
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-3:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
//           CHECK:   llvm.br
//   CHECK-COUNT-3:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//  CHECK-COUNT-32:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>, 3>
// CHECK-COUNT-128:   "llvm.intr.fmuladd"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//   CHECK-COUNT-3:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
//           CHECK:   llvm.br
//   CHECK-COUNT-3:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//  CHECK-COUNT-32:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>, 3>
// CHECK-COUNT-128:   "llvm.intr.fmuladd"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//   CHECK-COUNT-4:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>>

// -----

// Check that a generic op representing a matmul is getting the same
// configuration as the matmul op.
#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
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
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @dot_dispatch_0 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @dot_dispatch_0 layout(#executable_layout)
    builtin.module {
      func.func @dot_dispatch_0() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:1024x1024xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:1024x1024xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:1024x1024xf32>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:1024x1024xf32> -> tensor<1024x1024xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:1024x1024xf32> -> tensor<1024x1024xf32>
        %15 = linalg.init_tensor [1024, 1024] : tensor<1024x1024xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
        %17 = linalg.generic #matmul_trait 
            ins(%8, %10 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%16 : tensor<1024x1024xf32>)  {
          ^bb(%a: f32, %b: f32, %c: f32) :
            %d = arith.mulf %a, %b: f32
            %e = arith.addf %c, %d: f32
            linalg.yield %e : f32
          } -> (tensor<1024x1024xf32>)
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:1024x1024xf32>
        return
      }
    }
  }
}

//   CHECK-LABEL: hal.executable public @dot_dispatch_0
//         CHECK:   hal.executable.variant public @cuda
//         CHECK:   llvm.br
// CHECK-COUNT-8:   "llvm.intr.fmuladd"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//         CHECK:   llvm.br
// CHECK-COUNT-2:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>>

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @conv2d_dispatch_0 {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @conv2d_dispatch_0 layout(#executable_layout)
  builtin.module {
    func.func @conv2d_dispatch_0() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c1 = arith.constant 1 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:1x4x4x2xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:3x2x2x1xf32>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:1x2x3x1xf32>
      %11 = flow.dispatch.tensor.load %0, offsets = [0, 0 ,0, 0], sizes = [1, 4, 4, 2], strides = [1, 1, 1, 1]
          : !flow.dispatch.tensor<readonly:1x4x4x2xf32> -> tensor<1x4x4x2xf32>
      %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 2, 2, 1], strides = [1, 1, 1, 1]
          : !flow.dispatch.tensor<readonly:3x2x2x1xf32> -> tensor<3x2x2x1xf32>
      %20 = linalg.init_tensor [1, 2, 3, 1] : tensor<1x2x3x1xf32>
      %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x2x3x1xf32>) -> tensor<1x2x3x1xf32>
      %22 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
          ins(%11, %13 : tensor<1x4x4x2xf32>, tensor<3x2x2x1xf32>) outs(%21 : tensor<1x2x3x1xf32>) -> tensor<1x2x3x1xf32>
      flow.dispatch.tensor.store %22, %2, offsets = [0, 0, 0, 0], sizes = [1, 2, 3, 1], strides = [1, 1, 1, 1]
          : tensor<1x2x3x1xf32> -> !flow.dispatch.tensor<writeonly:1x2x3x1xf32>
      return
    }
  }
}
}

//   CHECK-LABEL: hal.executable public @conv2d_dispatch_0
//         CHECK:   hal.executable.variant public @cuda
// CHECK-COUNT-3:   llvm.load %{{.*}} : !llvm.ptr<f32>
//         CHECK:   lvm.fmul %{{.*}}, %{{.*}}  : f32
//         CHECK:   llvm.fadd %{{.*}}, %{{.*}}  : f32
//         CHECK:   llvm.store {{.*}} : !llvm.ptr<f32>

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @simpleMath_ex_dispatch_0 {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @add_dispatch_0 layout(#executable_layout)
  builtin.module {
    func.func @add_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:16xf32>
      %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:16xf32>
      %3 = linalg.init_tensor [16] : tensor<16xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %5 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
      } -> tensor<16xf32>
      flow.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
      return
    }
  }
}
}

// CHECK-LABEL: hal.executable public @simpleMath_ex_dispatch_0
//       CHECK:   hal.executable.variant public @cuda
//       CHECK:   llvm.mlir.global private constant @{{.*}}(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<16xf32>)
//       CHECK:   llvm.fadd

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @reduction_dispatch {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @reduction layout(#executable_layout)
  builtin.module {
    func.func @reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c96 = arith.constant 96 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:14x14x96xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:96xf32>
      %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [14, 14, 96], strides = [1, 1, 1]
          : !flow.dispatch.tensor<readonly:14x14x96xf32> -> tensor<14x14x96xf32>
      %8 = linalg.init_tensor [96] : tensor<96xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<96xf32>) -> tensor<96xf32>
      %10 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0)>],
            iterator_types = ["parallel", "reduction", "reduction"]}
            ins(%5 : tensor<14x14x96xf32>) outs(%9 : tensor<96xf32>) {
        ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
          %11 = arith.addf %arg1, %arg2 : f32
          linalg.yield %11 : f32
        } -> tensor<96xf32>
      flow.dispatch.tensor.store %10, %1, offsets = [0], sizes = [96], strides = [1]
          : tensor<96xf32> -> !flow.dispatch.tensor<writeonly:96xf32>
      return
    }
  }
}
}

// CHECK-LABEL: hal.executable public @reduction_dispatch
//       CHECK:   hal.executable.variant public @cuda
//       CHECK:   llvm.fadd

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @vector_add_dispatch {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @vector_add_dispatch layout(#executable_layout)
  builtin.module {
    func.func @vector_add_dispatch() {
      %c0 = arith.constant 0 : index
      %c16384 = arith.constant 16384 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:16384xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:16384xf32>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:16384xf32>
      %6 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [16384], strides = [1]
          : !flow.dispatch.tensor<readonly:16384xf32> -> tensor<16384xf32>
      %8 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [16384], strides = [1]
          : !flow.dispatch.tensor<readonly:16384xf32> -> tensor<16384xf32>
      %10 = linalg.init_tensor [16384] : tensor<16384xf32>
      %11 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}
          ins(%6, %8 : tensor<16384xf32>, tensor<16384xf32>) outs(%10 : tensor<16384xf32>) {
        ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
          %12 = arith.addf %arg1, %arg2 : f32
          linalg.yield %12 : f32
        } -> tensor<16384xf32>
      flow.dispatch.tensor.store %11, %2, offsets = [0], sizes = [16384], strides = [1]
          : tensor<16384xf32> -> !flow.dispatch.tensor<writeonly:16384xf32>
      return
    }
  }
}
}

//   CHECK-LABEL: hal.executable public @vector_add_dispatch
//         CHECK:   hal.executable.variant public @cuda
//         CHECK:   llvm.fadd %{{.*}}, %{{.*}}  : vector<4xf32
//         CHECK:   llvm.store %{{.*}} : !llvm.ptr<vector<4xf32>>

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 16384)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 16384, s0)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> (d0)>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @vector_reduction_dispatch {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @vector_reduction_dispatch layout(#executable_layout)
  builtin.module {
    func.func @vector_reduction_dispatch() {
      %c0 = arith.constant 0 : index
      %c16384 = arith.constant 16384 : index
      %cst = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:512x16384xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:16384xf32>
      %5 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 16384], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:512x16384xf32> -> tensor<512x16384xf32>
      %8 = linalg.init_tensor [16384] : tensor<16384xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<16384xf32>) -> tensor<16384xf32>
      %10 = linalg.generic {
          indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]}
          ins(%5 : tensor<512x16384xf32>) outs(%9 : tensor<16384xf32>) {
        ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
          %11 = arith.addf %arg1, %arg2 : f32
          linalg.yield %11 : f32
        } -> tensor<16384xf32>
      flow.dispatch.tensor.store %10, %1, offsets = [0], sizes = [16384], strides = [1]
          : tensor<16384xf32> -> !flow.dispatch.tensor<writeonly:16384xf32>
      return
    }
  }
}
}

//   CHECK-LABEL: hal.executable public @vector_reduction_dispatch
//         CHECK:   hal.executable.variant public @cuda
//         CHECK:   llvm.fadd %{{.*}}, %{{.*}} : vector<4xf32>
//         CHECK:   llvm.store %{{.*}} : !llvm.ptr<vector<4xf32>>

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @mma_fused {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.entry_point public @_large_aligned_dispatch_0 ordinal(0) layout(#hal.executable.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>, #hal.descriptor_set.binding<2, storage_buffer>]>]>)
  builtin.module {
    func.func @_large_aligned_dispatch_0() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c2048 = arith.constant 2048 : index
      %c512 = arith.constant 512 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:2048x1024xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:1024x512xf32>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:2048x512xf32>
      %di = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:2048x512xf32>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:2048x1024xf32> -> tensor<2048x1024xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:1024x512xf32> -> tensor<1024x512xf32>   
      %d = flow.dispatch.tensor.load %di, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:2048x512xf32> -> tensor<2048x512xf32>             
      %init = linalg.init_tensor [2048, 512] : tensor<2048x512xf32>
      %f = linalg.fill ins(%cst : f32) outs(%init : tensor<2048x512xf32>) -> tensor<2048x512xf32>
      %m = linalg.matmul ins(%3, %4 : tensor<2048x1024xf32>, tensor<1024x512xf32>) outs(%f : tensor<2048x512xf32>) -> tensor<2048x512xf32>
      %init2 = linalg.init_tensor [2048, 512] : tensor<2048x512xf32>
      %a = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%m, %d : tensor<2048x512xf32>, tensor<2048x512xf32>) outs(%m : tensor<2048x512xf32>) {
        ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
          %19 = arith.addf %arg3, %arg4 : f32
          linalg.yield %19 : f32
        } -> (tensor<2048x512xf32>)
        flow.dispatch.tensor.store %a, %2, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1]
          : tensor<2048x512xf32> -> !flow.dispatch.tensor<writeonly:2048x512xf32>
      return
    }
  }
}
}

//     CHECK-LABEL: hal.executable public @mma_fused
//           CHECK:   hal.executable.variant public @cuda
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//           CHECK:   nvvm.cp.async.wait.group 0
//   CHECK-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECK-COUNT-2:   nvvm.wmma.mma
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//           CHECK:   nvvm.cp.async.wait.group 0
//   CHECK-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECK-COUNT-2:   nvvm.wmma.mma
//   CHECK-COUNT-8:   llvm.fadd
//   CHECK-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<f32>, f32, f32, f32, f32, f32, f32, f32, f32

// case with larger pipeline depth
//     CHECKP-LABEL: hal.executable public @mma_fused
//           CHECKP:   hal.executable.variant public @cuda
//       CHECKP-NOT:   llvm.store
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//           CHECKP:   llvm.br
//           CHECKP:   nvvm.cp.async.wait.group 3
//   CHECKP-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECKP-COUNT-2:   nvvm.wmma.mma
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//           CHECKP:   llvm.br
//           CHECKP:   nvvm.cp.async.wait.group 3
//   CHECKP-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECKP-COUNT-2:   nvvm.wmma.mma
//           CHECKP:   nvvm.cp.async.wait.group 2
//   CHECKP-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECKP-COUNT-2:   nvvm.wmma.mma
//           CHECKP:   nvvm.cp.async.wait.group 1
//   CHECKP-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECKP-COUNT-2:   nvvm.wmma.mma
//           CHECKP:   nvvm.cp.async.wait.group 0
//   CHECKP-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECKP-COUNT-2:   nvvm.wmma.mma
//   CHECKP-COUNT-8:   llvm.fadd
//   CHECKP-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<f32>, f32, f32, f32, f32, f32, f32, f32, f32

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @mma_fused_fp16 {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.entry_point public @_large_aligned_dispatch_0 ordinal(0) layout(#hal.executable.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>, #hal.descriptor_set.binding<2, storage_buffer>]>]>)
  builtin.module {
    func.func @_large_aligned_dispatch_0() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c512 = arith.constant 512 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:2048x1024xf16>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:1024x512xf16>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:2048x512xf16>
      %di = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:2048x512xf16>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:2048x1024xf16> -> tensor<2048x1024xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:1024x512xf16> -> tensor<1024x512xf16>   
      %d = flow.dispatch.tensor.load %di, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:2048x512xf16> -> tensor<2048x512xf16>             
      %init = linalg.init_tensor [2048, 512] : tensor<2048x512xf16>
      %f = linalg.fill ins(%cst : f16) outs(%init : tensor<2048x512xf16>) -> tensor<2048x512xf16>
      %m = linalg.matmul ins(%3, %4 : tensor<2048x1024xf16>, tensor<1024x512xf16>) outs(%f : tensor<2048x512xf16>) -> tensor<2048x512xf16>
      %init2 = linalg.init_tensor [2048, 512] : tensor<2048x512xf16>
      %a = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%m, %d : tensor<2048x512xf16>, tensor<2048x512xf16>) outs(%m : tensor<2048x512xf16>) {
        ^bb0(%arg3: f16, %arg4: f16, %arg5: f16):  // no predecessors
          %19 = arith.addf %arg3, %arg4 : f16
          linalg.yield %19 : f16
        } -> (tensor<2048x512xf16>)
        flow.dispatch.tensor.store %a, %2, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1]
          : tensor<2048x512xf16> -> !flow.dispatch.tensor<writeonly:2048x512xf16>
      return
    }
  }
}
}

//     CHECK-LABEL: hal.executable public @mma_fused_fp16
//           CHECK:   hal.executable.variant public @cuda
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//           CHECK:   nvvm.cp.async.wait.group 0
//   CHECK-COUNT-2:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f16, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)
//   CHECK-COUNT-1:   nvvm.wmma.mma
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//   CHECK-COUNT-2:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f16, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)
//   CHECK-COUNT-1:   nvvm.wmma.mma
//   CHECK-COUNT-4:   llvm.fadd
//   CHECK-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<f16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>

// case with larger pipeline depth
//     CHECKP-LABEL: hal.executable public @mma_fused_fp16
//           CHECKP:   hal.executable.variant public @cuda
//       CHECKP-NOT:   llvm.store
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//           CHECKP:   llvm.br
//           CHECKP:   nvvm.cp.async.wait.group 3
//   CHECKP-COUNT-2:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f16, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)
//   CHECKP-COUNT-1:   nvvm.wmma.mma
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//           CHECKP:   llvm.br
//           CHECKP:   nvvm.cp.async.wait.group 3
//   CHECKP-COUNT-2:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f16, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)
//   CHECKP-COUNT-1:   nvvm.wmma.mma
//           CHECKP:   nvvm.cp.async.wait.group 2
//   CHECKP-COUNT-2:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f16, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)
//   CHECKP-COUNT-1:   nvvm.wmma.mma
//           CHECKP:   nvvm.cp.async.wait.group 1
//   CHECKP-COUNT-2:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f16, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)
//   CHECKP-COUNT-1:   nvvm.wmma.mma
//           CHECKP:   nvvm.cp.async.wait.group 0
//   CHECKP-COUNT-2:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f16, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)
//   CHECKP-COUNT-1:   nvvm.wmma.mma
//   CHECKP-COUNT-4:   llvm.fadd
//   CHECKP-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<f16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 4)>
#map2 = affine_map<(d0)[s0] -> (s0, -d0 + 32)>
#map3 = affine_map<(d0)[s0] -> (s0, -d0 + 64)>
#map4 = affine_map<(d0)[s0] -> (-d0 + 4, s0)>
#map5 = affine_map<(d0)[s0] -> (-d0 + 32, s0)>
#map6 = affine_map<(d0)[s0] -> (-d0 + 64, s0)>
  hal.executable @large_dot_general_dispatch_0 {
    hal.executable.variant public @cuda, target = #executable_target_cuda_nvptx_fb {
      hal.executable.entry_point @large_dot_general_dispatch_0 layout(#executable_layout)
      builtin.module {
        func.func @large_dot_general_dispatch_0() {
          %c64 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %c4 = arith.constant 4 : index
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32)
              : !flow.dispatch.tensor<readonly:4x32x1024xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32)
              : !flow.dispatch.tensor<readonly:4x1024x64xf32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32)
              : !flow.dispatch.tensor<writeonly:4x32x64xf32>
          %11 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 32, 1024], strides = [1, 1, 1]
              : !flow.dispatch.tensor<readonly:4x32x1024xf32> -> tensor<4x32x1024xf32>
          %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4, 1024, 64], strides = [1, 1, 1]
              : !flow.dispatch.tensor<readonly:4x1024x64xf32> -> tensor<4x1024x64xf32>
          %17 = linalg.init_tensor [4, 32, 64] : tensor<4x32x64xf32>
          %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
          %19 = linalg.batch_matmul ins(%11, %13 : tensor<4x32x1024xf32>, tensor<4x1024x64xf32>)
              outs(%18 : tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
          flow.dispatch.tensor.store %19, %2, offsets = [0, 0, 0], sizes = [4, 32, 64], strides = [1, 1, 1]
              : tensor<4x32x64xf32> -> !flow.dispatch.tensor<writeonly:4x32x64xf32>
          return
        }
      }
    }
  }
//     CHECK-LABEL: hal.executable public @large_dot_general_dispatch_0
//           CHECK:   hal.executable.variant public @cuda
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//           CHECK:   nvvm.cp.async.wait.group 0
//   CHECK-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECK-COUNT-2:   nvvm.wmma.mma
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//           CHECK:   nvvm.cp.async.wait.group 0
//   CHECK-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECK-COUNT-2:   nvvm.wmma.mma
//   CHECK-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<f32>, f32, f32, f32, f32, f32, f32, f32, f32

// case with larger pipeline depth
//     CHECKP-LABEL: hal.executable public @large_dot_general_dispatch_0
//           CHECKP:   hal.executable.variant public @cuda
//       CHECKP-NOT:   llvm.store
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//           CHECKP:   llvm.br
//           CHECKP:   nvvm.cp.async.wait.group 3
//   CHECKP-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECKP-COUNT-2:   nvvm.wmma.mma
//   CHECKP-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECKP:   nvvm.cp.async.commit.group
//           CHECKP:   llvm.br
//           CHECKP:   nvvm.cp.async.wait.group 3
//   CHECKP-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECKP-COUNT-2:   nvvm.wmma.mma
//           CHECKP:   nvvm.cp.async.wait.group 2
//   CHECKP-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECKP-COUNT-2:   nvvm.wmma.mma
//           CHECKP:   nvvm.cp.async.wait.group 1
//   CHECKP-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECKP-COUNT-2:   nvvm.wmma.mma
//           CHECKP:   nvvm.cp.async.wait.group 0
//   CHECKP-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECKP-COUNT-2:   nvvm.wmma.mma
//   CHECKP-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<f32>, f32, f32, f32, f32, f32, f32, f32, f32

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
#map0 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
  hal.executable public @split_k_gemm {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.entry_point public @split_k_gemm ordinal(0) layout(#executable_layout)
      builtin.module {
        func.func @split_k_gemm() {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:2048x4x256xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:4x256x512xf32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:4x2048x512xf32>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2048, 4, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:2048x4x256xf32> -> tensor<2048x4x256xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4, 256, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:4x256x512xf32> -> tensor<4x256x512xf32>
          %5 = linalg.init_tensor [4, 2048, 512] : tensor<4x2048x512xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x2048x512xf32>) -> tensor<4x2048x512xf32>
          %7 = linalg.generic {indexing_maps = [#map0, #map1, #map2],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%3, %4 : tensor<2048x4x256xf32>, tensor<4x256x512xf32>)
          outs(%6 : tensor<4x2048x512xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %8 = arith.mulf %arg0, %arg1 : f32
            %9 = arith.addf %arg2, %8 : f32
            linalg.yield %9 : f32
          } -> tensor<4x2048x512xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [4, 2048, 512], strides = [1, 1, 1] : tensor<4x2048x512xf32> -> !flow.dispatch.tensor<writeonly:4x2048x512xf32>
          return
        }
      }
    }
  }
//     CHECK-LABEL: hal.executable public @split_k_gemm
//           CHECK:   hal.executable.variant public @cuda
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//           CHECK:   nvvm.cp.async.wait.group 0
//   CHECK-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECK-COUNT-2:   nvvm.wmma.mma
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//           CHECK:   nvvm.cp.async.wait.group 0
//   CHECK-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECK-COUNT-2:   nvvm.wmma.mma
//   CHECK-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<f32>, f32, f32, f32, f32, f32, f32, f32, f32
