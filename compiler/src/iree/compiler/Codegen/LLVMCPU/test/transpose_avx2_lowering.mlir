// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=false})))' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu",
  "embedded-elf-x86_64", {
    cpu_features = "+avx2",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 32 : index,
    target_triple = "x86_64-none-elf"
  }>


hal.executable private @transpose_10_8x8_pattern {
  hal.executable.variant @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export @transpose_10_8x8_pattern layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_10_8x8_pattern() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
        %6 = linalg.generic {
          indexing_maps = [ affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3 : tensor<512x1024xf32>) outs(%5 : tensor<1024x512xf32>) {
          ^bb0(%arg1: f32, %arg2: f32):
            linalg.yield %arg1 : f32
          } -> tensor<1024x512xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: func.func @transpose_10_8x8_pattern
//  CHECK-COUNT-8: vector.load
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
// CHECK-COUNT-12: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//  CHECK-COUNT-8: llvm.inline_asm asm_dialect {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
//  CHECK-COUNT-8: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
//  CHECK-COUNT-8: vector.store

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu",
  "embedded-elf-x86_64", {
    cpu_features = "+avx2",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 32 : index,
    target_triple = "x86_64-none-elf"
  }>


hal.executable private @transpose_021_8x8_pattern {
  hal.executable.variant @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export @transpose_021_8x8_pattern layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_021_8x8_pattern() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x96x128xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x128x96xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 96, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x96x128xf32>> -> tensor<64x96x128xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [64, 128, 96], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<64x128x96xf32>> -> tensor<64x128x96xf32>
        %6 = linalg.generic {
          indexing_maps = [ affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%3 : tensor<64x96x128xf32>) outs(%5 : tensor<64x128x96xf32>) {
          ^bb0(%arg1: f32, %arg2: f32):
            linalg.yield %arg1 : f32
          } -> tensor<64x128x96xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [64, 128, 96], strides = [1, 1, 1] : tensor<64x128x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x128x96xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: func.func @transpose_021_8x8_pattern
//  CHECK-COUNT-8: vector.load
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
// CHECK-COUNT-12: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//  CHECK-COUNT-8: llvm.inline_asm asm_dialect {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
//  CHECK-COUNT-8: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
//  CHECK-COUNT-8: vector.store

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu",
  "embedded-elf-x86_64", {
    cpu_features = "+avx2",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 32 : index,
    target_triple = "x86_64-none-elf"
  }>


hal.executable private @transpose_201_8x8_pattern {
  hal.executable.variant @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export @transpose_201_8x8_pattern layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_201_8x8_pattern() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x96x128xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x64x96xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 96, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x96x128xf32>> -> tensor<64x96x128xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [128, 64, 96], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<128x64x96xf32>> -> tensor<128x64x96xf32>
        %6 = linalg.generic {
          indexing_maps = [ affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%3 : tensor<64x96x128xf32>) outs(%5 : tensor<128x64x96xf32>) {
          ^bb0(%arg1: f32, %arg2: f32):
            linalg.yield %arg1 : f32
          } -> tensor<128x64x96xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [128, 64, 96], strides = [1, 1, 1] : tensor<128x64x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x64x96xf32>>
        return
      }
    }
  }
}

//   CHECK-LABEL: func.func @transpose_201_8x8_pattern
//  CHECK-COUNT-8: vector.load
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
// CHECK-COUNT-12: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//  CHECK-COUNT-8: llvm.inline_asm asm_dialect {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
//  CHECK-COUNT-8: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
//  CHECK-COUNT-8: vector.store

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu",
  "embedded-elf-x86_64", {
    cpu_features = "+avx2",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 32 : index,
    target_triple = "x86_64-none-elf"
  }>


hal.executable private @transpose_210_8x8_pattern {
  hal.executable.variant @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export @transpose_210_8x8_pattern layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_210_8x8_pattern() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x96x128xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x96x64xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 96, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x96x128xf32>> -> tensor<64x96x128xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [128, 96, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<128x96x64xf32>> -> tensor<128x96x64xf32>
        %6 = linalg.generic {
          indexing_maps = [ affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%3 : tensor<64x96x128xf32>) outs(%5 : tensor<128x96x64xf32>) {
          ^bb0(%arg1: f32, %arg2: f32):
            linalg.yield %arg1 : f32
          } -> tensor<128x96x64xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [128, 96, 64], strides = [1, 1, 1] : tensor<128x96x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x96x64xf32>>
        return
      }
    }
  }
}

//   CHECK-LABEL: func.func @transpose_210_8x8_pattern
//  CHECK-COUNT-8: vector.load
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
// CHECK-COUNT-12: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//  CHECK-COUNT-8: llvm.inline_asm asm_dialect {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
//  CHECK-COUNT-8: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
//  CHECK-COUNT-8: vector.store

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu",
  "embedded-elf-x86_64", {
    cpu_features = "+avx2",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 32 : index,
    target_triple = "x86_64-none-elf"
  }>


hal.executable private @transpose_120_8x8_pattern {
  hal.executable.variant @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export @transpose_120_8x8_pattern layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_120_8x8_pattern() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x96x128xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<96x128x64xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 96, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x96x128xf32>> -> tensor<64x96x128xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [96, 128, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<96x128x64xf32>> -> tensor<96x128x64xf32>
        %6 = linalg.generic {
          indexing_maps = [ affine_map<(d0, d1, d2) -> (d2, d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%3 : tensor<64x96x128xf32>) outs(%5 : tensor<96x128x64xf32>) {
          ^bb0(%arg1: f32, %arg2: f32):
            linalg.yield %arg1 : f32
          } -> tensor<96x128x64xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [96, 128, 64], strides = [1, 1, 1] : tensor<96x128x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<96x128x64xf32>>
        return
      }
    }
  }
}

//   CHECK-LABEL: func.func @transpose_120_8x8_pattern
//  CHECK-COUNT-8: vector.load
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
// CHECK-COUNT-12: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//  CHECK-COUNT-8: llvm.inline_asm asm_dialect {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
//  CHECK-COUNT-8: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
//  CHECK-COUNT-8: vector.store

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu",
  "embedded-elf-x86_64", {
    cpu_features = "+avx2",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 32 : index,
    target_triple = "x86_64-none-elf"
  }>


hal.executable private @transpose_102 {
  hal.executable.variant @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export @transpose_102 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_102() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x96x128xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<96x64x128xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 96, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x96x128xf32>> -> tensor<64x96x128xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [96, 64, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<96x64x128xf32>> -> tensor<96x64x128xf32>
        %6 = linalg.generic {
          indexing_maps = [ affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%3 : tensor<64x96x128xf32>) outs(%5 : tensor<96x64x128xf32>) {
          ^bb0(%arg1: f32, %arg2: f32):
            linalg.yield %arg1 : f32
          } -> tensor<96x64x128xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [96, 64, 128], strides = [1, 1, 1] : tensor<96x64x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<96x64x128xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @transpose_102
//   CHECK-NOT: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
//   CHECK-NOT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu",
  "embedded-elf-x86_64", {
    // No '+avx2' cpu feature.
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 32 : index,
    target_triple = "x86_64-none-elf"
  }>


hal.executable private @test_no_avx2_feature {
  hal.executable.variant @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export @test_no_avx2_feature layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @test_no_avx2_feature() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
        %6 = linalg.generic {
          indexing_maps = [ affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3 : tensor<512x1024xf32>) outs(%5 : tensor<1024x512xf32>) {
          ^bb0(%arg1: f32, %arg2: f32):
            linalg.yield %arg1 : f32
          } -> tensor<1024x512xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @test_no_avx2_feature
//   CHECK-NOT: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
//   CHECK-NOT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
