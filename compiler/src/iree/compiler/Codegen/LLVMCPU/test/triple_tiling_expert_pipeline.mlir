// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' --iree-llvmcpu-enable-triple-tiling-pipeline --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_128_384_1536  {
  hal.executable.variant @system_elf_x86_64, target = <
    "llvm-cpu",
    "embedded-elf-x86_64", {
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 4 : index,
      target_triple = "x86_64-unknown-unknown-eabi-elf"
    }> {
    hal.executable.export @matmul_128_384_1536 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_128_384_1536() {
        %c786432 = arith.constant 786432 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x1536xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c786432) : !flow.dispatch.tensor<readonly:tensor<1536x384xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1536xf32>> -> tensor<128x1536xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1536, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1536x384xf32>> -> tensor<1536x384xf32>
        %5 = tensor.empty() : tensor<128x384xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x384xf32>) -> tensor<128x384xf32>
        %7 = linalg.matmul ins(%3, %4 : tensor<128x1536xf32>, tensor<1536x384xf32>) outs(%6 : tensor<128x384xf32>) -> tensor<128x384xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<128x384xf32>) outs(%5 : tensor<128x384xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %9 = math.exp %arg0 : f32
          linalg.yield %9 : f32
        } -> tensor<128x384xf32>
        flow.dispatch.tensor.store %8, %2, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : tensor<128x384xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
        return
      }
    }
  }
}

// CHECK:     func.func @matmul_128_384_1536
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1536:.+]] = arith.constant 1536 : index
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           vector.store
// CHECK:       scf.for {{.+}} = %[[C0]] to %[[C1536]] step
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               vector.outerproduct
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           {{.+}} = math.exp {{.+}} : vector<{{.+}}>
