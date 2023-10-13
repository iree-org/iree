// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-linalg-to-nvvm-pipeline)))" -iree-codegen-llvmgpu-use-wmma %s | FileCheck %s

// Verify that a simple element wise op gets lowered succefully all the way to
// nvvm/llvm dialect.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @simpleMath_ex_dispatch_0 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.export @add_dispatch_0 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @add_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<16xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<16xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<16xf32>>
      %3 = tensor.empty() : tensor<16xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %5 = flow.dispatch.tensor.load %1, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16xf32>
        flow.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:tensor<16xf32>>
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
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @dot_dispatch_0 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @dot_dispatch_0 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dot_dispatch_0() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
        %15 = tensor.empty() : tensor<1024x1024xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
        %17 = linalg.matmul ins(%8, %10 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
            outs(%16 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
        return
      }
    }
  }
}

//     CHECK-LABEL: hal.executable public @dot_dispatch_0
//           CHECK:   hal.executable.variant public @cuda
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-3:   llvm.load {{.*}} : !llvm.ptr<1> -> vector<4xf32>
//           CHECK:   llvm.br
//   CHECK-COUNT-3:   llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<3>
//  CHECK-COUNT-32:   llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
// CHECK-COUNT-128:   llvm.intr.fmuladd({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//   CHECK-COUNT-3:   llvm.load {{.*}} : !llvm.ptr<1> -> vector<4xf32>
//           CHECK:   llvm.br
//   CHECK-COUNT-3:   llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<3>
//  CHECK-COUNT-32:   llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
// CHECK-COUNT-128:   llvm.intr.fmuladd({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//   CHECK-COUNT-4:   llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<1>

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
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @dot_dispatch_0 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @dot_dispatch_0 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dot_dispatch_0() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
        %15 = tensor.empty() : tensor<1024x1024xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
        %17 = linalg.generic #matmul_trait
            ins(%8, %10 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%16 : tensor<1024x1024xf32>)  {
          ^bb(%a: f32, %b: f32, %c: f32) :
            %d = arith.mulf %a, %b: f32
            %e = arith.addf %c, %d: f32
            linalg.yield %e : f32
          } -> (tensor<1024x1024xf32>)
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
        return
      }
    }
  }
}

//   CHECK-LABEL: hal.executable public @dot_dispatch_0
//         CHECK:   hal.executable.variant public @cuda
//         CHECK:   llvm.br
// CHECK-COUNT-8:   llvm.intr.fmuladd({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//         CHECK:   llvm.br
// CHECK-COUNT-2:   llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<1>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @conv2d_dispatch_0 {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.export @conv2d_dispatch_0 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index, %arg7 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @conv2d_dispatch_0() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c1 = arith.constant 1 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x4x4x2xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x2x2x1xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x2x3x1xf32>>
      %11 = flow.dispatch.tensor.load %0, offsets = [0, 0 ,0, 0], sizes = [1, 4, 4, 2], strides = [1, 1, 1, 1]
          : !flow.dispatch.tensor<readonly:tensor<1x4x4x2xf32>> -> tensor<1x4x4x2xf32>
      %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 2, 2, 1], strides = [1, 1, 1, 1]
          : !flow.dispatch.tensor<readonly:tensor<3x2x2x1xf32>> -> tensor<3x2x2x1xf32>
      %20 = tensor.empty() : tensor<1x2x3x1xf32>
      %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x2x3x1xf32>) -> tensor<1x2x3x1xf32>
      %22 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
          ins(%11, %13 : tensor<1x4x4x2xf32>, tensor<3x2x2x1xf32>) outs(%21 : tensor<1x2x3x1xf32>) -> tensor<1x2x3x1xf32>
      flow.dispatch.tensor.store %22, %2, offsets = [0, 0, 0, 0], sizes = [1, 2, 3, 1], strides = [1, 1, 1, 1]
          : tensor<1x2x3x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x2x3x1xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: hal.executable public @conv2d_dispatch_0
//         CHECK:   hal.executable.variant public @cuda
// CHECK-COUNT-3:   llvm.load %{{.*}} : !llvm.ptr<1> -> f32
//         CHECK:   lvm.fmul %{{.*}}, %{{.*}}  : f32
//         CHECK:   llvm.fadd %{{.*}}, %{{.*}}  : f32
//         CHECK:   llvm.store {{.*}} : f32, !llvm.ptr<1>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @simpleMath_ex_dispatch_0 {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.export @add_dispatch_0 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @add_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<16xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<16xf32>>
      %3 = tensor.empty() : tensor<16xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %5 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
      } -> tensor<16xf32>
      flow.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:tensor<16xf32>>
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @reduction_dispatch {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.export @reduction layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c96 = arith.constant 96 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<14x14x96xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<96xf32>>
      %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [14, 14, 96], strides = [1, 1, 1]
          : !flow.dispatch.tensor<readonly:tensor<14x14x96xf32>> -> tensor<14x14x96xf32>
      %8 = tensor.empty() : tensor<96xf32>
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
          : tensor<96xf32> -> !flow.dispatch.tensor<writeonly:tensor<96xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: hal.executable public @reduction_dispatch
//       CHECK:   hal.executable.variant public @cuda
//       CHECK:   llvm.fadd

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @vector_add_dispatch {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.export @vector_add_dispatch layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @vector_add_dispatch() {
      %c0 = arith.constant 0 : index
      %c16384 = arith.constant 16384 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<16384xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<16384xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<16384xf32>>
      %6 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [16384], strides = [1]
          : !flow.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
      %8 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [16384], strides = [1]
          : !flow.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
      %10 = tensor.empty() : tensor<16384xf32>
      %11 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}
          ins(%6, %8 : tensor<16384xf32>, tensor<16384xf32>) outs(%10 : tensor<16384xf32>) {
        ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
          %12 = arith.addf %arg1, %arg2 : f32
          linalg.yield %12 : f32
        } -> tensor<16384xf32>
      flow.dispatch.tensor.store %11, %2, offsets = [0], sizes = [16384], strides = [1]
          : tensor<16384xf32> -> !flow.dispatch.tensor<writeonly:tensor<16384xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: hal.executable public @vector_add_dispatch
//         CHECK:   hal.executable.variant public @cuda
//         CHECK:   llvm.fadd %{{.*}}, %{{.*}}  : vector<4xf32
//         CHECK:   llvm.store %{{.*}} : vector<4xf32>, !llvm.ptr<1>

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 16384)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 16384, s0)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @vector_reduction_dispatch {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.export @vector_reduction_dispatch layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @vector_reduction_dispatch() {
      %c0 = arith.constant 0 : index
      %c16384 = arith.constant 16384 : index
      %cst = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x16384xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<16384xf32>>
      %5 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 16384], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<512x16384xf32>> -> tensor<512x16384xf32>
      %8 = tensor.empty() : tensor<16384xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<16384xf32>) -> tensor<16384xf32>
      %10 = linalg.generic {
          indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]}
          ins(%5 : tensor<512x16384xf32>) outs(%9 : tensor<16384xf32>) {
        ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
          %11 = arith.addf %arg1, %arg2 : f32
          linalg.yield %11 : f32
        } -> tensor<16384xf32>
      flow.dispatch.tensor.store %10, %1, offsets = [0], sizes = [16384], strides = [1]
          : tensor<16384xf32> -> !flow.dispatch.tensor<writeonly:tensor<16384xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: hal.executable public @vector_reduction_dispatch
//         CHECK:   hal.executable.variant public @cuda
// CHECK-COUNT-5:     nvvm.shfl.sync  bfly

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @mma_fused {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.export public @_large_aligned_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>, #hal.descriptor_set.binding<2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @_large_aligned_dispatch_0() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c2048 = arith.constant 2048 : index
      %c512 = arith.constant 512 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<2048x1024xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
      %di = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<2048x512xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<2048x1024xf32>> -> tensor<2048x1024xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
      %d = flow.dispatch.tensor.load %di, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<2048x512xf32>> -> tensor<2048x512xf32>
      %init = tensor.empty() : tensor<2048x512xf32>
      %f = linalg.fill ins(%cst : f32) outs(%init : tensor<2048x512xf32>) -> tensor<2048x512xf32>
      %m = linalg.matmul ins(%3, %4 : tensor<2048x1024xf32>, tensor<1024x512xf32>) outs(%f : tensor<2048x512xf32>) -> tensor<2048x512xf32>
      %init2 = tensor.empty() : tensor<2048x512xf32>
      %a = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%m, %d : tensor<2048x512xf32>, tensor<2048x512xf32>) outs(%init2 : tensor<2048x512xf32>) {
        ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
          %19 = arith.addf %arg3, %arg4 : f32
          linalg.yield %19 : f32
        } -> (tensor<2048x512xf32>)
        flow.dispatch.tensor.store %a, %2, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1]
          : tensor<2048x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
      return
    }
  }
}
}

// case with larger pipeline depth
//     CHECK-LABEL: hal.executable public @mma_fused
//           CHECK:   hal.executable.variant public @cuda
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//           CHECK:   nvvm.cp.async.wait.group 3
//   CHECK-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECK-COUNT-2:   nvvm.wmma.mma
//   CHECK-COUNT-2:   llvm.inline_asm has_side_effects asm_dialect = att "cp.async.cg.shared.global [$0], [$1], $2, $3;\0A", "r,l,n,r" {{.*}}, {{.*}}, {{.*}}, {{.*}} : (!llvm.ptr<3>, !llvm.ptr<1>, i32, i32) -> ()
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//       CHECK-NOT:   nvvm.wmma.mma
//   CHECK-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<3>, f32, f32, f32, f32, f32, f32, f32, f32
//           CHECK:   vvm.barrier0
//           CHECK:   llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
//           CHECK:   llvm.fadd {{.*}} : vector<4xf32>
//           CHECK:   llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<1>
//           CHECK:   llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
//           CHECK:   llvm.fadd {{.*}} : vector<4xf32>
//           CHECK:   llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<1>



// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @mma_fused_fp16 {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.export public @_large_aligned_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>, #hal.descriptor_set.binding<2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @_large_aligned_dispatch_0() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c512 = arith.constant 512 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<2048x1024xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1024x512xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<2048x512xf16>>
      %di = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<2048x512xf16>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<2048x1024xf16>> -> tensor<2048x1024xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<1024x512xf16>> -> tensor<1024x512xf16>
      %d = flow.dispatch.tensor.load %di, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<2048x512xf16>> -> tensor<2048x512xf16>
      %init = tensor.empty() : tensor<2048x512xf16>
      %f = linalg.fill ins(%cst : f16) outs(%init : tensor<2048x512xf16>) -> tensor<2048x512xf16>
      %m = linalg.matmul ins(%3, %4 : tensor<2048x1024xf16>, tensor<1024x512xf16>) outs(%f : tensor<2048x512xf16>) -> tensor<2048x512xf16>
      %init2 = tensor.empty() : tensor<2048x512xf16>
      %a = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%m, %d : tensor<2048x512xf16>, tensor<2048x512xf16>) outs(%init2 : tensor<2048x512xf16>) {
        ^bb0(%arg3: f16, %arg4: f16, %arg5: f16):  // no predecessors
          %19 = arith.addf %arg3, %arg4 : f16
          linalg.yield %19 : f16
        } -> (tensor<2048x512xf16>)
        flow.dispatch.tensor.store %a, %2, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1]
          : tensor<2048x512xf16> -> !flow.dispatch.tensor<writeonly:tensor<2048x512xf16>>
      return
    }
  }
}
}

// case with larger pipeline depth
//     CHECK-LABEL: hal.executable public @mma_fused_fp16
//           CHECK:   hal.executable.variant public @cuda
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//           CHECK:   nvvm.cp.async.wait.group 3
//   CHECK-COUNT-2:   nvvm.wmma.load{{.*}} : (!llvm.ptr<3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)
//   CHECK-COUNT-1:   nvvm.wmma.mma
//   CHECK-COUNT-2:   llvm.inline_asm has_side_effects asm_dialect = att "cp.async.cg.shared.global [$0], [$1], $2, $3;\0A", "r,l,n,r" {{.*}}, {{.*}}, {{.*}}, {{.*}} : (!llvm.ptr<3>, !llvm.ptr<1>, i32, i32) -> ()
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//       CHECK-NOT:   nvvm.wmma.mma
//   CHECK-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<3>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>
//           CHECK:   vvm.barrier0
//           CHECK:   llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf16>
//           CHECK:   llvm.fadd {{.*}} : vector<8xf16>
//           CHECK:   llvm.store {{.*}} : vector<8xf16>, !llvm.ptr<1>
//           CHECK:   vvm.barrier0

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
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
      hal.executable.export @large_dot_general_dispatch_0 layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 :index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @large_dot_general_dispatch_0() {
          %c64 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %c4 = arith.constant 4 : index
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0)
              : !flow.dispatch.tensor<readonly:tensor<4x32x1024xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0)
              : !flow.dispatch.tensor<readonly:tensor<4x1024x64xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0)
              : !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>>
          %11 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 32, 1024], strides = [1, 1, 1]
              : !flow.dispatch.tensor<readonly:tensor<4x32x1024xf32>> -> tensor<4x32x1024xf32>
          %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4, 1024, 64], strides = [1, 1, 1]
              : !flow.dispatch.tensor<readonly:tensor<4x1024x64xf32>> -> tensor<4x1024x64xf32>
          %17 = tensor.empty() : tensor<4x32x64xf32>
          %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
          %19 = linalg.batch_matmul ins(%11, %13 : tensor<4x32x1024xf32>, tensor<4x1024x64xf32>)
              outs(%18 : tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
          flow.dispatch.tensor.store %19, %2, offsets = [0, 0, 0], sizes = [4, 32, 64], strides = [1, 1, 1]
              : tensor<4x32x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>>
          return
        }
      }
    }
  }

// case with larger pipeline depth
//     CHECK-LABEL: hal.executable public @large_dot_general_dispatch_0
//           CHECK:   hal.executable.variant public @cuda
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//   CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//           CHECK:   nvvm.cp.async.wait.group 3
//   CHECK-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECK-COUNT-2:   nvvm.wmma.mma
//   CHECK-COUNT-2:   llvm.inline_asm has_side_effects asm_dialect = att "cp.async.cg.shared.global [$0], [$1], $2, $3;\0A", "r,l,n,r" {{.*}}, {{.*}}, {{.*}}, {{.*}} : (!llvm.ptr<3>, !llvm.ptr<1>, i32, i32) -> ()
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//       CHECK-NOT:   nvvm.wmma.mma
//   CHECK-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<3>, f32, f32, f32, f32, f32, f32, f32, f32
//           CHECK:   vvm.barrier0
//           CHECK:   llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
//           CHECK:   llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<1>
//           CHECK:   llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
//           CHECK:   llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<1>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
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
      hal.executable.export public @split_k_gemm ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @split_k_gemm() {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x4x256xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x256x512xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x2048x512xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2048, 4, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x4x256xf32>> -> tensor<2048x4x256xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4, 256, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x256x512xf32>> -> tensor<4x256x512xf32>
          %5 = tensor.empty() : tensor<4x2048x512xf32>
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
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [4, 2048, 512], strides = [1, 1, 1] : tensor<4x2048x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x2048x512xf32>>
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
//           CHECK:   nvvm.cp.async.wait.group 3
//   CHECK-COUNT-4:   nvvm.wmma.load{{.*}} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)
//   CHECK-COUNT-2:   nvvm.wmma.mma
//           CHECK:   llvm.inline_asm has_side_effects asm_dialect = att "cp.async.cg.shared.global [$0], [$1], $2, $3;\0A", "r,l,n,r" {{.*}}, {{.*}}, {{.*}}, {{.*}} : (!llvm.ptr<3>, !llvm.ptr<1>, i32, i32) -> ()
//           CHECK:   nvvm.cp.async.commit.group
//           CHECK:   llvm.br
//       CHECK-NOT:   nvvm.wmma.mma
//   CHECK-COUNT-1:   nvvm.wmma.store {{.*}} : !llvm.ptr<3>, f32, f32, f32, f32, f32, f32, f32, f32
//           CHECK:   vvm.barrier0
//           CHECK:   llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
//           CHECK:   llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<1>
//           CHECK:   llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xf32>
//           CHECK:   llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<1>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
  hal.executable public @pooling_dynamic {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.export public @pooling_dynamic ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 :index, %arg4 : index, %arg5 : index, %arg6 : index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @pooling_dynamic() {
          %c1_i64 = arith.constant 1 : i64
          %c2_i64 = arith.constant 2 : i64
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.constant.load[0] : i32
          %s = arith.index_cast %0 : i32 to index
          %14 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%s) : !flow.dispatch.tensor<readonly:tensor<?x2048x?x?xf32>>{%s, %s, %s}
          %15 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%s) : !flow.dispatch.tensor<writeonly:tensor<?x2048x1x1xf32>>{%s}
          %16 = flow.dispatch.tensor.load %14, offsets = [0, 0, 0, 0], sizes = [%s, 2048, %s, %s], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x2048x?x?xf32>>{%s, %s, %s} -> tensor<?x2048x?x?xf32>
          %19 = tensor.empty(%s) : tensor<?x2048x1x1xf32>
          %38 = tensor.empty(%s, %s) : tensor<?x?xf32>
          %39 = linalg.fill ins(%cst : f32) outs(%19 : tensor<?x2048x1x1xf32>) -> tensor<?x2048x1x1xf32>
          %40 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%16, %38 : tensor<?x2048x?x?xf32>, tensor<?x?xf32>) outs(%39 : tensor<?x2048x1x1xf32>) -> tensor<?x2048x1x1xf32>
          flow.dispatch.tensor.store %40, %15, offsets = [0, 0, 0, 0], sizes = [%s, 2048, 1, 1], strides = [1, 1, 1, 1] : tensor<?x2048x1x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x2048x1x1xf32>>{%s}
          return
        }
      }
    }
  }

// Just check that compilation succeed.
//     CHECK-LABEL: hal.executable public @pooling_dynamic
//           CHECK:   hal.executable.variant public @cuda

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 16384)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 16384, s0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @warp_reduction_dispatch {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.export @warp_reduction_dispatch layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @warp_reduction_dispatch() {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %cst = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<512xf32>>
      %5 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
      %8 = tensor.empty() : tensor<512xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<512xf32>) -> tensor<512xf32>
      %10 = linalg.generic {
          indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]}
          ins(%5 : tensor<512x1024xf32>) outs(%9 : tensor<512xf32>) {
        ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
          %11 = arith.addf %arg1, %arg2 : f32
          linalg.yield %11 : f32
        } -> tensor<512xf32>
      flow.dispatch.tensor.store %10, %1, offsets = [0], sizes = [512], strides = [1]
          : tensor<512xf32> -> !flow.dispatch.tensor<writeonly:tensor<512xf32>>
      return
    }
  }
}
}

// Check that we generate a warp reduce code sequence.
//   CHECK-LABEL: hal.executable public @warp_reduction_dispatch
//         CHECK:   hal.executable.variant public @cuda
// CHECK-COUNT-5:     nvvm.shfl.sync  bfly
//         CHECK:     llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<3>
//         CHECK:     nvvm.barrier0
//         CHECK:     llvm.load {{.*}} : !llvm.ptr<3> -> f32
// CHECK-COUNT-3:     nvvm.shfl.sync  bfly

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @warp_reduction_broadcast_dispatch {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.export @warp_reduction_broadcast_dispatch layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @warp_reduction_broadcast_dispatch() {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %cst_0 = arith.constant 3.840000e+02 : f32
      %cst = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<512x1024xf32>>
      %5 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
      %8 = tensor.empty() : tensor<512xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<512xf32>) -> tensor<512xf32>
      %10 = linalg.generic {
          indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]}
          ins(%5 : tensor<512x1024xf32>) outs(%9 : tensor<512xf32>) {
        ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
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
      flow.dispatch.tensor.store %11, %1, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1]
          : tensor<512x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<512x1024xf32>>
      return
    }
  }
}
}

// Check that we generate a group reduce fused with broadcast + elementwise.
//   CHECK-LABEL: hal.executable public @warp_reduction_broadcast_dispatch
//         CHECK:   hal.executable.variant public @cuda
// CHECK-COUNT-5:     nvvm.shfl.sync  bfly
//         CHECK:     llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<3>
//         CHECK:     nvvm.barrier0
//         CHECK:     llvm.load {{.*}} : !llvm.ptr<3> -> f32
// CHECK-COUNT-3:     nvvm.shfl.sync  bfly
//         CHECK:     llvm.fdiv %{{.*}}, %{{.*}} 
//         CHECK:     llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<1>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @shared_mem_alloc {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
    hal.executable.export public @shared_mem_alloc ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @shared_mem_alloc() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0xFF800000> : tensor<14x14x480xf32>
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<29x29x480xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<14x14x480xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [29, 29, 480], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<29x29x480xf32>> -> tensor<29x29x480xf32>
        %3 = tensor.empty() : tensor<3x3xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%2, %3 : tensor<29x29x480xf32>, tensor<3x3xf32>) outs(%cst : tensor<14x14x480xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %5 = arith.maximumf %arg2, %arg0 : f32
          linalg.yield %5 : f32
        } -> tensor<14x14x480xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [14, 14, 480], strides = [1, 1, 1] : tensor<14x14x480xf32> -> !flow.dispatch.tensor<writeonly:tensor<14x14x480xf32>>
        return
      }
    }
  }
}

// Check that bufferization is emitting correct code for the temp shared
// memory alloc.
//   CHECK-LABEL: hal.executable private @shared_mem_alloc
//         CHECK:   hal.executable.variant public @cuda
//         CHECK:     nvvm.barrier0
//         CHECK:     llvm.store %{{.*}}, %{{.*}} : f32, !llvm.ptr<3>
//         CHECK:     nvvm.barrier0
//         CHECK:     nvvm.barrier0
//         CHECK:     llvm.load %{{.*}} : !llvm.ptr<3> -> f32
//         CHECK:     nvvm.barrier0

// -----


#config = #iree_codegen.lowering_config<tile_sizes = [[32,32]]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
hal.executable private @shared_mem_transpose  {
  hal.executable.variant @cuda, target = #executable_target_cuda_nvptx_fb {
    hal.executable.export @shared_mem_transpose layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
        hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
        func.func @shared_mem_transpose() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x768xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<768x2048xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x768xf32>> -> tensor<2048x768xf32>
          %3 = tensor.empty() : tensor<768x2048xf32>
          %4 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<2048x768xf32>) outs(%3 : tensor<768x2048xf32>) {
          ^bb0(%arg0: f32, %arg1: f32):
            linalg.yield %arg0 : f32
          } -> tensor<768x2048xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [768, 2048], strides = [1, 1] : tensor<768x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<768x2048xf32>>
          return
        }
    }
  }
}

// Check that bufferization is emitting correct code for the temp shared
// memory alloc.
//   CHECK-LABEL: hal.executable private @shared_mem_transpose
//         CHECK:   hal.executable.variant public @cuda
//         CHECK:     nvvm.barrier0
//         CHECK:     llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> vector<4xf32>
//         CHECK:     llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<3>
//         CHECK:     nvvm.barrier0
