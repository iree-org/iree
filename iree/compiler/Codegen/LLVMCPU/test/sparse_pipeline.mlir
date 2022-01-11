// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-codegen-linalg-to-llvm-pipeline))' %s | IreeFileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable @sparse_dispatch {
hal.executable.variant public @embedded_elf_x86_64, target = <"llvm", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}> {
  hal.executable.entry_point public @_large_dispatch_0 ordinal(0) layout(#hal.executable.layout<push_constants = 1, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>, #hal.descriptor_set.binding<2, storage_buffer>]>]>) {
  ^bb0(%arg0: index, %arg1: index, %arg2: index):  // no predecessors
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module  {
    func @_large_dispatch_0() {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.constant.load[0] values([0 : index]) : index
      %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:8x8xf32>
      %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%0) alignment(32) : !flow.dispatch.tensor<readonly:8xf32>
      %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:8xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:8x8xf32> -> tensor<8x8xf32>
      %5 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [8], strides = [1] : !flow.dispatch.tensor<readonly:8xf32> -> tensor<8xf32>
      %6 = linalg.init_tensor [8] : tensor<8xf32>
      %7 = sparse_tensor.convert %4 : tensor<8x8xf32> to tensor<8x8xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>>
      %8 = linalg.fill(%cst, %6) : f32, tensor<8xf32> -> tensor<8xf32>
      %9 = linalg.generic {doc = "X(i) += A(i,j) * B(j)", indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%7, %5 : tensor<8x8xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>>, tensor<8xf32>) outs(%8 : tensor<8xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
        %10 = arith.mulf %arg0, %arg1 : f32
        %11 = arith.addf %arg2, %10 : f32
        linalg.yield %11 : f32
      } -> tensor<8xf32>
      flow.dispatch.tensor.store %9, %3, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:8xf32>
      return
    }
  }
}
}

//     CHECK-LABEL: hal.executable public @sparse_dispatch
//           CHECK:  %[[SB0:.+]] = llvm.call @newSparseTensor
//           CHECK:  llvm.call @addEltF32(%[[SB0]]
//           CHECK:  %[[SB1:.+]] = llvm.call @newSparseTensor({{.*}}, %[[SB0:.+]])
//           CHECK:  %{{.*}} = llvm.call @sparsePointers(%[[SB1]], %{{.*}}) : (!llvm.ptr<i8>, i64) -> !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
//           CHECK:  %{{.*}} = llvm.call @sparseIndices(%[[SB1]], %{{.*}}) : (!llvm.ptr<i8>, i64) -> !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
//           CHECK:  %{{.*}} = llvm.call @sparseValuesF32(%[[SB1]]) : (!llvm.ptr<i8>) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//           CHECK:     %{{.*}} = llvm.fmul %{{.*}}, %{{.*}} : f32
//           CHECK:     %{{.*}} = llvm.fadd %{{.*}}, %{{.*}} : f32
