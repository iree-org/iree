// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-llvmgpu-configuration-pipeline, iree-codegen-linalg-to-rocdl-pipeline)))" %s | FileCheck %s

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
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
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
//       CHECK:   hal.executable.variant public @rocm
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
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
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

//   CHECK-LABEL: hal.executable public @dot_dispatch_0
//         CHECK:   hal.executable.variant public @rocm
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
