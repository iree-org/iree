// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx908 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline)))" %s | FileCheck %s --check-prefix=CDNA1
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline)))" %s | FileCheck %s --check-prefix=CDNA3
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline)))" %s | FileCheck %s --check-prefix=RDNA3

// Verify that a simple element wise op gets lowered successfully all the way to
// nvvm/llvm dialect.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @simpleMath_ex_dispatch_0 {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @add_dispatch_0 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root(%arg1)
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @add_dispatch_0() {
      %c0 = arith.constant 0 : index
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
    }
  }
}

// CDNA1-LABEL: hal.executable public @simpleMath_ex_dispatch_0
//       CDNA1:   hal.executable.variant public @rocm
//       CDNA1:   llvm.fadd

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @dot_dispatch_0 {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @dot_dispatch_0 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root(%arg1, %arg2, %arg3)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dot_dispatch_0() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
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
    }
  }
}

//   RDNA3-LABEL: hal.executable public @dot_dispatch_0
//         RDNA3:   hal.executable.variant public @rocm
//       RDNA3-NOT:   llvm.store
//           RDNA3:   llvm.br
//   RDNA3-COUNT-1:    llvm.load {{.*}} : !llvm.ptr<7> -> vector<32xf32>
//  RDNA3-COUNT-32:    llvm.load {{.*}} : !llvm.ptr<7> -> vector<16xf32>
//  RDNA3-COUNT-32:    llvm.intr.fmuladd({{.*}}) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
//   RDNA3-COUNT-1:    llvm.store {{.*}} : vector<16xf32>, !llvm.ptr<7>
//           RDNA3:   llvm.br

// -----

// Verify that the ceildivsi op gets expanded and lowered successfully all the way to
// the llvm dialect.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @ceildiv_expand_dispatch {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export public @ceildiv_expand layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root(%arg1)
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @ceildiv_expand() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
      %3 = tensor.empty() : tensor<16xi32>
      %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets=[0], sizes=[16], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
      %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets=[0], sizes=[16], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xi32>, tensor<16xi32>) outs(%3 : tensor<16xi32>) {
      ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
          %7 = arith.ceildivsi %arg0, %arg1 : i32
          linalg.yield %7 : i32
        } -> tensor<16xi32>
        iree_tensor_ext.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16], strides=[1] : tensor<16xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
        return
      }
    }
  }
}

//   CDNA3-LABEL: hal.executable public @ceildiv_expand_dispatch
//         CDNA3:   hal.executable.variant public @rocm
//     CDNA3-NOT:     arith.ceildivsi
// CDNA3-COUNT-1:     llvm.sdiv {{.*}} : vector<1xi32>
// CDNA3-COUNT-1:     llvm.mul {{.*}} : vector<1xi32>
// CDNA3-COUNT-3:     llvm.icmp {{.*}} : vector<1xi32>
// CHECK-COUNT-1:     llvm.icmp {{.*}} : vector<1xi1>
// CDNA3-COUNT-1:     llvm.and {{.*}} : vector<1xi1>
// CDNA3-COUNT-1:     llvm.add {{.*}} : vector<1xi32>
// CDNA3-COUNT-1:     llvm.select {{.*}} : vector<1xi1>, vector<1xi32>
