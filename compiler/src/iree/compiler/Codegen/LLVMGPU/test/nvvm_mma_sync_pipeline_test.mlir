// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-linalg-to-nvvm-pipeline)))" -iree-codegen-llvmgpu-use-mma-sync %s | FileCheck %s

// Verify that a simple element wise op gets lowered succefully all the way to
// nvvm/llvm dialect via mma.sync path.

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

// mma.sync.16816.f16.f16 / TensorCore(f16):
//    CHECK-LABEL: hal.executable public @mma_fused_fp16
//          CHECK:   hal.executable.variant public @cuda
//      CHECK-NOT:   llvm.store
//  CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//          CHECK:   nvvm.cp.async.commit.group
//  CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//          CHECK:   nvvm.cp.async.commit.group
//  CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//          CHECK:   nvvm.cp.async.commit.group
//          CHECK:   nvvm.cp.async.wait.group 2
//  CHECK-COUNT-2:   nvvm.ldmatrix {{.*}} : (!llvm.ptr<f16, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
//          CHECK:   llvm.br
//  CHECK-COUNT-2:   nvvm.ldmatrix {{.*}} : (!llvm.ptr<f16, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
//  CHECK-COUNT-2:   nvvm.mma.sync {{.*}} {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
//  CHECK-COUNT-2:   llvm.inline_asm has_side_effects asm_dialect = att "cp.async.cg.shared.global [$0], [$1], $2, $3;\0A", "r,l,n,r" {{.*}}, {{.*}}, {{.*}}, {{.*}} : (!llvm.ptr<i8, 3>, !llvm.ptr<i8, 1>, i32, i32) -> !llvm.void
//          CHECK:   nvvm.cp.async.commit.group
//          CHECK:   nvvm.cp.async.wait.group 2
//  CHECK-COUNT-2:   nvvm.ldmatrix {{.*}} : (!llvm.ptr<f16, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
//  CHECK-COUNT-2:   nvvm.mma.sync {{.*}} {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
//          CHECK:   llvm.br
//      CHECK-NOT:   nvvm.mma.sync
//  CHECK-COUNT-4:   llvm.store {{.*}} : !llvm.ptr<vector<2xf16>, 3>
//          CHECK:   llvm.load {{.*}} : !llvm.ptr<vector<8xf16>, 3>
//          CHECK:   llvm.store {{.*}} : !llvm.ptr<vector<8xf16>>

// -----


#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @mma_fused_f32 {
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

// mma.sync.1688.f32.tf32 / TensorCore(f32):
//    CHECK-LABEL: hal.executable public @mma_fused_f32
//          CHECK:   hal.executable.variant public @cuda
//      CHECK-NOT:   llvm.store
//  CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//          CHECK:   nvvm.cp.async.commit.group
//  CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//          CHECK:   nvvm.cp.async.commit.group
//  CHECK-COUNT-2:   nvvm.cp.async.shared.global {{.*}}, {{.*}}, 16
//          CHECK:   nvvm.cp.async.commit.group
//          CHECK:   nvvm.cp.async.wait.group 2
//  CHECK-COUNT-1:   nvvm.ldmatrix{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
//  CHECK-COUNT-4:   llvm.extractvalue{{.*}} : !llvm.struct<(i32, i32, i32, i32)> 
//          CHECK:   llvm.br
//  CHECK-COUNT-1:   nvvm.ldmatrix{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
//  CHECK-COUNT-4:   llvm.extractvalue{{.*}} : !llvm.struct<(i32, i32, i32, i32)> 
//  CHECK-COUNT-2:   nvvm.mma.sync {{.*}} {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<tf32>, multiplicandBPtxType = #nvvm.mma_type<tf32>, shape = #nvvm.shape<m = 16, n = 8, k = 8>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
//  CHECK-COUNT-2:   llvm.inline_asm has_side_effects asm_dialect = att "cp.async.cg.shared.global [$0], [$1], $2, $3;\0A", "r,l,n,r" {{.*}}, {{.*}}, {{.*}}, {{.*}} : (!llvm.ptr<i8, 3>, !llvm.ptr<i8, 1>, i32, i32) -> !llvm.void
//          CHECK:   nvvm.cp.async.commit.group
//          CHECK:   nvvm.cp.async.wait.group 2
//  CHECK-COUNT-1:   nvvm.ldmatrix{{.*}} : (!llvm.ptr<f32, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
//  CHECK-COUNT-4:   llvm.extractvalue{{.*}} : !llvm.struct<(i32, i32, i32, i32)> 
//  CHECK-COUNT-2:   nvvm.mma.sync {{.*}} {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<tf32>, multiplicandBPtxType = #nvvm.mma_type<tf32>, shape = #nvvm.shape<m = 16, n = 8, k = 8>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
//          CHECK:   llvm.br
//      CHECK-NOT:   nvvm.mma.sync
//  CHECK-COUNT-4:   llvm.store {{.*}} : !llvm.ptr<vector<2xf32>, 3>
//    CHECK-COUNT:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//    CHECK-COUNT:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>>
//    CHECK-COUNT:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//    CHECK-COUNT:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>>
//    CHECK-COUNT:   nvvm.barrier0
//    CHECK-COUNT:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//    CHECK-COUNT:   llvm.fadd {{.*}} : vector<4xf32>
//    CHECK-COUNT:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>>
//    CHECK-COUNT:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//    CHECK-COUNT:   llvm.fadd {{.*}} : vector<4xf32>
//    CHECK-COUNT:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>>