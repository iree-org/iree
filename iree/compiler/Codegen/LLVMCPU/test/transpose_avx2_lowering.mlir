// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=false}))' -split-input-file %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm",
  "embedded-elf-x86_64", {
    cpu_features = "+avx2",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 32 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }>


hal.executable private @test_8x8_avx2_pattern {
  hal.executable.variant @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.entry_point @test_8x8_avx2_pattern layout(#executable_layout)
    builtin.module {
      func @test_8x8_avx2_pattern() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:512x1024xf32>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:1024x512xf32>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:512x1024xf32> -> tensor<512x1024xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !flow.dispatch.tensor<writeonly:1024x512xf32> -> tensor<1024x512xf32>
        %6 = linalg.generic {
          indexing_maps = [ affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3 : tensor<512x1024xf32>) outs(%5 : tensor<1024x512xf32>) {
          ^bb0(%arg1: f32, %arg2: f32):
            linalg.yield %arg1 : f32
          } -> tensor<1024x512xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xf32> -> !flow.dispatch.tensor<writeonly:1024x512xf32>
        return
      }
    }
  }
}

// CHECK: func @test_8x8_avx2_pattern
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
// CHECK: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm",
  "embedded-elf-x86_64", {
    // No '+avx2' cpu feature.
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 32 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }>


hal.executable private @test_no_avx2_feature {
  hal.executable.variant @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.entry_point @test_no_avx2_feature layout(#executable_layout)
    builtin.module {
      func @test_no_avx2_feature() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:512x1024xf32>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:1024x512xf32>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:512x1024xf32> -> tensor<512x1024xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !flow.dispatch.tensor<writeonly:1024x512xf32> -> tensor<1024x512xf32>
        %6 = linalg.generic {
          indexing_maps = [ affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3 : tensor<512x1024xf32>) outs(%5 : tensor<1024x512xf32>) {
          ^bb0(%arg1: f32, %arg2: f32):
            linalg.yield %arg1 : f32
          } -> tensor<1024x512xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xf32> -> !flow.dispatch.tensor<writeonly:1024x512xf32>
        return
      }
    }
  }
}

//     CHECK: func @test_no_avx2_feature
// CHECK-NOT: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
// CHECK-NOT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
