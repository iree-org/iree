// Test un-peeled epilogue generating AsyncCopyOp using zfill
// RUN: iree-opt --iree-gpu-pipelining=epilogue-peeling=false --split-input-file %s | FileCheck %s
// RUN: iree-opt --iree-gpu-pipelining="pipeline-depth=3 schedule-index=2 epilogue-peeling=false" --split-input-file %s | FileCheck -check-prefix=CHECK-NV %s


func.func @_matmul_f16_f16_dispatch_0_fill_3456x1024() {
  %c2048 = arith.constant 2048 : index
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf16, "COp">
  %1 = gpu.thread_id  x
  %2 = gpu.thread_id  y
  %3 = gpu.thread_id  z
  %4 = memref.alloc() : memref<4x32x40xf16, 3>
  %5 = memref.alloc() : memref<4x32x40xf16, 3>
  %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<3456x2048xf16>
  memref.assume_alignment %6, 64 : memref<3456x2048xf16>
  %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<2048x1024xf16>
  memref.assume_alignment %7, 64 : memref<2048x1024xf16>
  %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<3456x1024xf16>
  memref.assume_alignment %8, 64 : memref<3456x1024xf16>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %9 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%1, %2, %3]
  %10 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%1]
  %11 = scf.for %arg0 = %c0 to %c2048 step %c32 iter_args(%arg1 = %0) -> (!gpu.mma_matrix<16x16xf16, "COp">) {
    gpu.barrier
    %14 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8 - (s1 floordiv 4) * 32)>()[%arg0, %1]
    %15 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 16 + s2 * 32 + s3 * 32 + s0 floordiv 4)>()[%1, %2, %3, %workgroup_id_y]
    %16 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) mod 4)>(%arg0)
    %17 = nvgpu.device_async_copy %6[%15, %14], %4[%16, %9, %10], 8 : memref<3456x2048xf16> to memref<4x32x40xf16, 3>
    %18 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 16 + s3 * 32 + s1 floordiv 4)>()[%arg0, %1, %2, %3]
    %19 = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 32 - (s0 floordiv 4) * 32)>()[%1, %workgroup_id_x]
    %20 = nvgpu.device_async_copy %7[%18, %19], %5[%16, %9, %10], 8 : memref<2048x1024xf16> to memref<4x32x40xf16, 3>
    %21 = nvgpu.device_async_create_group %17, %20
    nvgpu.device_async_wait %21
    gpu.barrier
    %22 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%2]
    %23 = gpu.subgroup_mma_load_matrix %4[%16, %22, %c0] {leadDimension = 40 : index} : memref<4x32x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %24 = gpu.subgroup_mma_load_matrix %4[%16, %22, %c16] {leadDimension = 40 : index} : memref<4x32x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %25 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 16)>()[%1]
    %26 = gpu.subgroup_mma_load_matrix %5[%16, %c0, %25] {leadDimension = 40 : index} : memref<4x32x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
    %27 = gpu.subgroup_mma_load_matrix %5[%16, %c16, %25] {leadDimension = 40 : index} : memref<4x32x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
    %28 = gpu.subgroup_mma_compute %23, %26, %arg1 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
    %29 = gpu.subgroup_mma_compute %24, %27, %28 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
    scf.yield %29 : !gpu.mma_matrix<16x16xf16, "COp">
  }
  %12 = affine.apply affine_map<()[s0, s1] -> (s0 * 16 + s1 * 32)>()[%2, %workgroup_id_y]
  %13 = affine.apply affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 32) * 16)>()[%1, %workgroup_id_x]
  gpu.subgroup_mma_store_matrix %11, %8[%12, %13] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<3456x1024xf16>
  return
}
// CHECK-LABEL: func.func @_matmul_f16_f16_dispatch_0_fill_3456x1024
// CHECK:  %[[CP_ID:.*]] = nvgpu.device_async_copy %[[GMEMPTR:.*]][%[[IDX:.*]]%[[IDY:.*]]], %[[SMEMPTR:.*]][%[[IDK_S:.*]]%[[IDX_S:.*]]%[[IDY_S:.*]]], 8, %[[PRED:.*]] : memref<3456x2048xf16> to memref<4x32x40xf16, 3>

// -----

func.func @nvidia_tenscore_schedule() {
  %c3 = arith.constant 3 : index
  %c31 = arith.constant 31 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant dense<0.000000e+00> : vector<2x2xf16>
  %c1280 = arith.constant 1280 : index
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = gpu.thread_id  x
  %1 = gpu.thread_id  y
  %2 = gpu.thread_id  z
  %alloc = memref.alloc() : memref<128x256xf16, #gpu.address_space<workgroup>>
  %alloc_1 = memref.alloc() : memref<3x128x32xf16, #gpu.address_space<workgroup>>
  %alloc_2 = memref.alloc() : memref<3x32x256xf16, #gpu.address_space<workgroup>>
  %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<512x1280xf16>
  memref.assume_alignment %3, 64 : memref<512x1280xf16>
  %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<1280x1280xf16>
  memref.assume_alignment %4, 64 : memref<1280x1280xf16>
  %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<512x1280xf16>
  memref.assume_alignment %5, 64 : memref<512x1280xf16>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %6:32 = scf.for %arg0 = %c0 to %c1280 step %c32 iter_args(%arg1 = %cst, %arg2 = %cst, %arg3 = %cst, %arg4 = %cst, %arg5 = %cst, %arg6 = %cst, %arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %cst, %arg30 = %cst, %arg31 = %cst, %arg32 = %cst) -> (vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>) {
    gpu.barrier
    %138 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8 - (s1 floordiv 4) * 32)>()[%arg0, %0]
    %139 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 32 + s2 * 64 + s3 * 128 + s0 floordiv 4)>()[%0, %1, %2, %workgroup_id_y]
    %140 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 4)>()[%0, %1, %2]
    %141 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
    %142 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) mod 3)>(%arg0)
    %143 = arith.andi %140, %c6 : index
    %144 = arith.shli %143, %c2 : index
    %145 = arith.xori %141, %144 : index
    %146 = nvgpu.device_async_copy %3[%139, %138], %alloc_1[%142, %140, %145], 8 {bypassL1} : memref<512x1280xf16> to memref<3x128x32xf16, #gpu.address_space<workgroup>>
    %147 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 32 + s2 * 64 + s3 * 128 + s0 floordiv 4 + 64)>()[%0, %1, %2, %workgroup_id_y]
    %148 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 4 + 64)>()[%0, %1, %2]
    %149 = arith.andi %148, %c6 : index
    %150 = arith.shli %149, %c2 : index
    %151 = arith.xori %141, %150 : index
    %152 = nvgpu.device_async_copy %3[%147, %138], %alloc_1[%142, %148, %151], 8 {bypassL1} : memref<512x1280xf16> to memref<3x128x32xf16, #gpu.address_space<workgroup>>
    %153 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 4 + s3 * 8 + s1 floordiv 32)>()[%arg0, %0, %1, %2]
    %154 = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 256 - (s0 floordiv 32) * 256)>()[%0, %workgroup_id_x]
    %155 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32)>()[%0, %1, %2]
    %156 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 32) * 256)>()[%0]
    %157 = arith.andi %155, %c31 : index
    %158 = arith.shli %157, %c3 : index
    %159 = arith.xori %156, %158 : index
    %160 = nvgpu.device_async_copy %4[%153, %154], %alloc_2[%142, %155, %159], 8 {bypassL1} : memref<1280x1280xf16> to memref<3x32x256xf16, #gpu.address_space<workgroup>>
    %161 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 4 + s3 * 8 + s1 floordiv 32 + 8)>()[%arg0, %0, %1, %2]
    %162 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 8)>()[%0, %1, %2]
    %163 = arith.andi %162, %c31 : index
    %164 = arith.shli %163, %c3 : index
    %165 = arith.xori %156, %164 : index
    %166 = nvgpu.device_async_copy %4[%161, %154], %alloc_2[%142, %162, %165], 8 {bypassL1} : memref<1280x1280xf16> to memref<3x32x256xf16, #gpu.address_space<workgroup>>
    %167 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 4 + s3 * 8 + s1 floordiv 32 + 16)>()[%arg0, %0, %1, %2]
    %168 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 16)>()[%0, %1, %2]
    %169 = arith.andi %168, %c31 : index
    %170 = arith.shli %169, %c3 : index
    %171 = arith.xori %156, %170 : index
    %172 = nvgpu.device_async_copy %4[%167, %154], %alloc_2[%142, %168, %171], 8 {bypassL1} : memref<1280x1280xf16> to memref<3x32x256xf16, #gpu.address_space<workgroup>>
    %173 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 4 + s3 * 8 + s1 floordiv 32 + 24)>()[%arg0, %0, %1, %2]
    %174 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 24)>()[%0, %1, %2]
    %175 = arith.andi %174, %c31 : index
    %176 = arith.shli %175, %c3 : index
    %177 = arith.xori %156, %176 : index
    %178 = nvgpu.device_async_copy %4[%173, %154], %alloc_2[%142, %174, %177], 8 {bypassL1} : memref<1280x1280xf16> to memref<3x32x256xf16, #gpu.address_space<workgroup>>
    %179 = nvgpu.device_async_create_group %146, %152, %160, %166, %172, %178
    nvgpu.device_async_wait %179
    gpu.barrier
    %180 = gpu.lane_id
    %181 = affine.apply affine_map<(d0)[s0] -> (d0 + s0 * 64 - (d0 floordiv 16) * 16)>(%180)[%1]
    %182 = affine.apply affine_map<(d0) -> ((d0 floordiv 16) * 8)>(%180)
    %183 = arith.andi %181, %c6 : index
    %184 = arith.shli %183, %c2 : index
    %185 = arith.xori %182, %184 : index
    %186 = nvgpu.ldmatrix %alloc_1[%142, %181, %185] {numTiles = 4 : i32, transpose = false} : memref<3x128x32xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %187 = affine.apply affine_map<(d0) -> ((d0 floordiv 16) * 8 + 16)>(%180)
    %188 = arith.xori %187, %184 : index
    %189 = nvgpu.ldmatrix %alloc_1[%142, %181, %188] {numTiles = 4 : i32, transpose = false} : memref<3x128x32xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %190 = affine.apply affine_map<(d0)[s0] -> (d0 + s0 * 64 - (d0 floordiv 16) * 16 + 16)>(%180)[%1]
    %191 = arith.andi %190, %c6 : index
    %192 = arith.shli %191, %c2 : index
    %193 = arith.xori %182, %192 : index
    %194 = nvgpu.ldmatrix %alloc_1[%142, %190, %193] {numTiles = 4 : i32, transpose = false} : memref<3x128x32xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %195 = arith.xori %187, %192 : index
    %196 = nvgpu.ldmatrix %alloc_1[%142, %190, %195] {numTiles = 4 : i32, transpose = false} : memref<3x128x32xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %197 = affine.apply affine_map<(d0)[s0] -> (d0 + s0 * 64 - (d0 floordiv 16) * 16 + 32)>(%180)[%1]
    %198 = arith.andi %197, %c6 : index
    %199 = arith.shli %198, %c2 : index
    %200 = arith.xori %182, %199 : index
    %201 = nvgpu.ldmatrix %alloc_1[%142, %197, %200] {numTiles = 4 : i32, transpose = false} : memref<3x128x32xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %202 = arith.xori %187, %199 : index
    %203 = nvgpu.ldmatrix %alloc_1[%142, %197, %202] {numTiles = 4 : i32, transpose = false} : memref<3x128x32xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %204 = affine.apply affine_map<(d0)[s0] -> (d0 + s0 * 64 - (d0 floordiv 16) * 16 + 48)>(%180)[%1]
    %205 = arith.andi %204, %c6 : index
    %206 = arith.shli %205, %c2 : index
    %207 = arith.xori %182, %206 : index
    %208 = nvgpu.ldmatrix %alloc_1[%142, %204, %207] {numTiles = 4 : i32, transpose = false} : memref<3x128x32xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %209 = arith.xori %187, %206 : index
    %210 = nvgpu.ldmatrix %alloc_1[%142, %204, %209] {numTiles = 4 : i32, transpose = false} : memref<3x128x32xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %211 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 16) * 8 + (s0 floordiv 32) * 64)>(%180)[%0]
    %212 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%180)
    %213 = arith.andi %212, %c31 : index
    %214 = arith.shli %213, %c3 : index
    %215 = arith.xori %211, %214 : index
    %216 = nvgpu.ldmatrix %alloc_2[%142, %212, %215] {numTiles = 4 : i32, transpose = true} : memref<3x32x256xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %217 = affine.apply affine_map<(d0) -> (d0 mod 16 + 16)>(%180)
    %218 = arith.andi %217, %c31 : index
    %219 = arith.shli %218, %c3 : index
    %220 = arith.xori %211, %219 : index
    %221 = nvgpu.ldmatrix %alloc_2[%142, %217, %220] {numTiles = 4 : i32, transpose = true} : memref<3x32x256xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %222 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 16) * 8 + (s0 floordiv 32) * 64 + 16)>(%180)[%0]
    %223 = arith.xori %222, %214 : index
    %224 = nvgpu.ldmatrix %alloc_2[%142, %212, %223] {numTiles = 4 : i32, transpose = true} : memref<3x32x256xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %225 = arith.xori %222, %219 : index
    %226 = nvgpu.ldmatrix %alloc_2[%142, %217, %225] {numTiles = 4 : i32, transpose = true} : memref<3x32x256xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %227 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 16) * 8 + (s0 floordiv 32) * 64 + 32)>(%180)[%0]
    %228 = arith.xori %227, %214 : index
    %229 = nvgpu.ldmatrix %alloc_2[%142, %212, %228] {numTiles = 4 : i32, transpose = true} : memref<3x32x256xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %230 = arith.xori %227, %219 : index
    %231 = nvgpu.ldmatrix %alloc_2[%142, %217, %230] {numTiles = 4 : i32, transpose = true} : memref<3x32x256xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %232 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 16) * 8 + (s0 floordiv 32) * 64 + 48)>(%180)[%0]
    %233 = arith.xori %232, %214 : index
    %234 = nvgpu.ldmatrix %alloc_2[%142, %212, %233] {numTiles = 4 : i32, transpose = true} : memref<3x32x256xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %235 = arith.xori %232, %219 : index
    %236 = nvgpu.ldmatrix %alloc_2[%142, %217, %235] {numTiles = 4 : i32, transpose = true} : memref<3x32x256xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
    %237 = vector.extract_strided_slice %216 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %238 = nvgpu.mma.sync(%186, %237, %arg1) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %239 = vector.extract_strided_slice %216 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %240 = nvgpu.mma.sync(%186, %239, %arg2) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %241 = vector.extract_strided_slice %224 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %242 = nvgpu.mma.sync(%186, %241, %arg3) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %243 = vector.extract_strided_slice %224 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %244 = nvgpu.mma.sync(%186, %243, %arg4) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %245 = vector.extract_strided_slice %229 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %246 = nvgpu.mma.sync(%186, %245, %arg5) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %247 = vector.extract_strided_slice %229 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %248 = nvgpu.mma.sync(%186, %247, %arg6) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %249 = vector.extract_strided_slice %234 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %250 = nvgpu.mma.sync(%186, %249, %arg7) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %251 = vector.extract_strided_slice %234 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %252 = nvgpu.mma.sync(%186, %251, %arg8) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %253 = nvgpu.mma.sync(%194, %237, %arg9) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %254 = nvgpu.mma.sync(%194, %239, %arg10) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %255 = nvgpu.mma.sync(%194, %241, %arg11) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %256 = nvgpu.mma.sync(%194, %243, %arg12) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %257 = nvgpu.mma.sync(%194, %245, %arg13) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %258 = nvgpu.mma.sync(%194, %247, %arg14) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %259 = nvgpu.mma.sync(%194, %249, %arg15) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %260 = nvgpu.mma.sync(%194, %251, %arg16) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %261 = nvgpu.mma.sync(%201, %237, %arg17) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %262 = nvgpu.mma.sync(%201, %239, %arg18) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %263 = nvgpu.mma.sync(%201, %241, %arg19) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %264 = nvgpu.mma.sync(%201, %243, %arg20) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %265 = nvgpu.mma.sync(%201, %245, %arg21) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %266 = nvgpu.mma.sync(%201, %247, %arg22) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %267 = nvgpu.mma.sync(%201, %249, %arg23) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %268 = nvgpu.mma.sync(%201, %251, %arg24) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %269 = nvgpu.mma.sync(%208, %237, %arg25) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %270 = nvgpu.mma.sync(%208, %239, %arg26) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %271 = nvgpu.mma.sync(%208, %241, %arg27) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %272 = nvgpu.mma.sync(%208, %243, %arg28) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %273 = nvgpu.mma.sync(%208, %245, %arg29) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %274 = nvgpu.mma.sync(%208, %247, %arg30) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %275 = nvgpu.mma.sync(%208, %249, %arg31) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %276 = nvgpu.mma.sync(%208, %251, %arg32) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %277 = vector.extract_strided_slice %221 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %278 = nvgpu.mma.sync(%189, %277, %238) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %279 = vector.extract_strided_slice %221 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %280 = nvgpu.mma.sync(%189, %279, %240) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %281 = vector.extract_strided_slice %226 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %282 = nvgpu.mma.sync(%189, %281, %242) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %283 = vector.extract_strided_slice %226 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %284 = nvgpu.mma.sync(%189, %283, %244) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %285 = vector.extract_strided_slice %231 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %286 = nvgpu.mma.sync(%189, %285, %246) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %287 = vector.extract_strided_slice %231 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %288 = nvgpu.mma.sync(%189, %287, %248) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %289 = vector.extract_strided_slice %236 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %290 = nvgpu.mma.sync(%189, %289, %250) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %291 = vector.extract_strided_slice %236 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
    %292 = nvgpu.mma.sync(%189, %291, %252) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %293 = nvgpu.mma.sync(%196, %277, %253) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %294 = nvgpu.mma.sync(%196, %279, %254) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %295 = nvgpu.mma.sync(%196, %281, %255) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %296 = nvgpu.mma.sync(%196, %283, %256) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %297 = nvgpu.mma.sync(%196, %285, %257) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %298 = nvgpu.mma.sync(%196, %287, %258) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %299 = nvgpu.mma.sync(%196, %289, %259) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %300 = nvgpu.mma.sync(%196, %291, %260) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %301 = nvgpu.mma.sync(%203, %277, %261) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %302 = nvgpu.mma.sync(%203, %279, %262) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %303 = nvgpu.mma.sync(%203, %281, %263) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %304 = nvgpu.mma.sync(%203, %283, %264) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %305 = nvgpu.mma.sync(%203, %285, %265) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %306 = nvgpu.mma.sync(%203, %287, %266) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %307 = nvgpu.mma.sync(%203, %289, %267) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %308 = nvgpu.mma.sync(%203, %291, %268) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %309 = nvgpu.mma.sync(%210, %277, %269) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %310 = nvgpu.mma.sync(%210, %279, %270) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %311 = nvgpu.mma.sync(%210, %281, %271) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %312 = nvgpu.mma.sync(%210, %283, %272) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %313 = nvgpu.mma.sync(%210, %285, %273) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %314 = nvgpu.mma.sync(%210, %287, %274) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %315 = nvgpu.mma.sync(%210, %289, %275) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    %316 = nvgpu.mma.sync(%210, %291, %276) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
    scf.yield %278, %280, %282, %284, %286, %288, %290, %292, %293, %294, %295, %296, %297, %298, %299, %300, %301, %302, %303, %304, %305, %306, %307, %308, %309, %310, %311, %312, %313, %314, %315, %316 : vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>
  }
  %7 = gpu.lane_id
  %8 = vector.extract %6#31[0] : vector<2x2xf16>
  %9 = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 floordiv 4 + 48)>()[%1, %7]
  %10 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 - (s1 floordiv 4) * 8 + (s0 floordiv 32) * 64 + 56)>()[%0, %7]
  vector.store %8, %alloc[%9, %10] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %11 = vector.extract %6#31[1] : vector<2x2xf16>
  %12 = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 floordiv 4 + 56)>()[%1, %7]
  vector.store %11, %alloc[%12, %10] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %13 = vector.extract %6#30[0] : vector<2x2xf16>
  %14 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 - (s1 floordiv 4) * 8 + (s0 floordiv 32) * 64 + 48)>()[%0, %7]
  vector.store %13, %alloc[%9, %14] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %15 = vector.extract %6#30[1] : vector<2x2xf16>
  vector.store %15, %alloc[%12, %14] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %16 = vector.extract %6#29[0] : vector<2x2xf16>
  %17 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 - (s1 floordiv 4) * 8 + (s0 floordiv 32) * 64 + 40)>()[%0, %7]
  vector.store %16, %alloc[%9, %17] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %18 = vector.extract %6#29[1] : vector<2x2xf16>
  vector.store %18, %alloc[%12, %17] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %19 = vector.extract %6#28[0] : vector<2x2xf16>
  %20 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 - (s1 floordiv 4) * 8 + (s0 floordiv 32) * 64 + 32)>()[%0, %7]
  vector.store %19, %alloc[%9, %20] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %21 = vector.extract %6#28[1] : vector<2x2xf16>
  vector.store %21, %alloc[%12, %20] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %22 = vector.extract %6#27[0] : vector<2x2xf16>
  %23 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 - (s1 floordiv 4) * 8 + (s0 floordiv 32) * 64 + 24)>()[%0, %7]
  vector.store %22, %alloc[%9, %23] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %24 = vector.extract %6#27[1] : vector<2x2xf16>
  vector.store %24, %alloc[%12, %23] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %25 = vector.extract %6#26[0] : vector<2x2xf16>
  %26 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 - (s1 floordiv 4) * 8 + (s0 floordiv 32) * 64 + 16)>()[%0, %7]
  vector.store %25, %alloc[%9, %26] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %27 = vector.extract %6#26[1] : vector<2x2xf16>
  vector.store %27, %alloc[%12, %26] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %28 = vector.extract %6#25[0] : vector<2x2xf16>
  %29 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 - (s1 floordiv 4) * 8 + (s0 floordiv 32) * 64 + 8)>()[%0, %7]
  vector.store %28, %alloc[%9, %29] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %30 = vector.extract %6#25[1] : vector<2x2xf16>
  vector.store %30, %alloc[%12, %29] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %31 = vector.extract %6#24[0] : vector<2x2xf16>
  %32 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 - (s1 floordiv 4) * 8 + (s0 floordiv 32) * 64)>()[%0, %7]
  vector.store %31, %alloc[%9, %32] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %33 = vector.extract %6#24[1] : vector<2x2xf16>
  vector.store %33, %alloc[%12, %32] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %34 = vector.extract %6#23[0] : vector<2x2xf16>
  %35 = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 floordiv 4 + 32)>()[%1, %7]
  vector.store %34, %alloc[%35, %10] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %36 = vector.extract %6#23[1] : vector<2x2xf16>
  %37 = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 floordiv 4 + 40)>()[%1, %7]
  vector.store %36, %alloc[%37, %10] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %38 = vector.extract %6#22[0] : vector<2x2xf16>
  vector.store %38, %alloc[%35, %14] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %39 = vector.extract %6#22[1] : vector<2x2xf16>
  vector.store %39, %alloc[%37, %14] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %40 = vector.extract %6#21[0] : vector<2x2xf16>
  vector.store %40, %alloc[%35, %17] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %41 = vector.extract %6#21[1] : vector<2x2xf16>
  vector.store %41, %alloc[%37, %17] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %42 = vector.extract %6#20[0] : vector<2x2xf16>
  vector.store %42, %alloc[%35, %20] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %43 = vector.extract %6#20[1] : vector<2x2xf16>
  vector.store %43, %alloc[%37, %20] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %44 = vector.extract %6#19[0] : vector<2x2xf16>
  vector.store %44, %alloc[%35, %23] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %45 = vector.extract %6#19[1] : vector<2x2xf16>
  vector.store %45, %alloc[%37, %23] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %46 = vector.extract %6#18[0] : vector<2x2xf16>
  vector.store %46, %alloc[%35, %26] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %47 = vector.extract %6#18[1] : vector<2x2xf16>
  vector.store %47, %alloc[%37, %26] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %48 = vector.extract %6#17[0] : vector<2x2xf16>
  vector.store %48, %alloc[%35, %29] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %49 = vector.extract %6#17[1] : vector<2x2xf16>
  vector.store %49, %alloc[%37, %29] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %50 = vector.extract %6#16[0] : vector<2x2xf16>
  vector.store %50, %alloc[%35, %32] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %51 = vector.extract %6#16[1] : vector<2x2xf16>
  vector.store %51, %alloc[%37, %32] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %52 = vector.extract %6#15[0] : vector<2x2xf16>
  %53 = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 floordiv 4 + 16)>()[%1, %7]
  vector.store %52, %alloc[%53, %10] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %54 = vector.extract %6#15[1] : vector<2x2xf16>
  %55 = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 floordiv 4 + 24)>()[%1, %7]
  vector.store %54, %alloc[%55, %10] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %56 = vector.extract %6#14[0] : vector<2x2xf16>
  vector.store %56, %alloc[%53, %14] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %57 = vector.extract %6#14[1] : vector<2x2xf16>
  vector.store %57, %alloc[%55, %14] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %58 = vector.extract %6#13[0] : vector<2x2xf16>
  vector.store %58, %alloc[%53, %17] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %59 = vector.extract %6#13[1] : vector<2x2xf16>
  vector.store %59, %alloc[%55, %17] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %60 = vector.extract %6#12[0] : vector<2x2xf16>
  vector.store %60, %alloc[%53, %20] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %61 = vector.extract %6#12[1] : vector<2x2xf16>
  vector.store %61, %alloc[%55, %20] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %62 = vector.extract %6#11[0] : vector<2x2xf16>
  vector.store %62, %alloc[%53, %23] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %63 = vector.extract %6#11[1] : vector<2x2xf16>
  vector.store %63, %alloc[%55, %23] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %64 = vector.extract %6#10[0] : vector<2x2xf16>
  vector.store %64, %alloc[%53, %26] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %65 = vector.extract %6#10[1] : vector<2x2xf16>
  vector.store %65, %alloc[%55, %26] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %66 = vector.extract %6#9[0] : vector<2x2xf16>
  vector.store %66, %alloc[%53, %29] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %67 = vector.extract %6#9[1] : vector<2x2xf16>
  vector.store %67, %alloc[%55, %29] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %68 = vector.extract %6#8[0] : vector<2x2xf16>
  vector.store %68, %alloc[%53, %32] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %69 = vector.extract %6#8[1] : vector<2x2xf16>
  vector.store %69, %alloc[%55, %32] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %70 = vector.extract %6#7[0] : vector<2x2xf16>
  %71 = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 floordiv 4)>()[%1, %7]
  vector.store %70, %alloc[%71, %10] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %72 = vector.extract %6#7[1] : vector<2x2xf16>
  %73 = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 floordiv 4 + 8)>()[%1, %7]
  vector.store %72, %alloc[%73, %10] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %74 = vector.extract %6#6[0] : vector<2x2xf16>
  vector.store %74, %alloc[%71, %14] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %75 = vector.extract %6#6[1] : vector<2x2xf16>
  vector.store %75, %alloc[%73, %14] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %76 = vector.extract %6#5[0] : vector<2x2xf16>
  vector.store %76, %alloc[%71, %17] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %77 = vector.extract %6#5[1] : vector<2x2xf16>
  vector.store %77, %alloc[%73, %17] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %78 = vector.extract %6#4[0] : vector<2x2xf16>
  vector.store %78, %alloc[%71, %20] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %79 = vector.extract %6#4[1] : vector<2x2xf16>
  vector.store %79, %alloc[%73, %20] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %80 = vector.extract %6#3[0] : vector<2x2xf16>
  vector.store %80, %alloc[%71, %23] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %81 = vector.extract %6#3[1] : vector<2x2xf16>
  vector.store %81, %alloc[%73, %23] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %82 = vector.extract %6#2[0] : vector<2x2xf16>
  vector.store %82, %alloc[%71, %26] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %83 = vector.extract %6#2[1] : vector<2x2xf16>
  vector.store %83, %alloc[%73, %26] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %84 = vector.extract %6#1[0] : vector<2x2xf16>
  vector.store %84, %alloc[%71, %29] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %85 = vector.extract %6#1[1] : vector<2x2xf16>
  vector.store %85, %alloc[%73, %29] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %86 = vector.extract %6#0[0] : vector<2x2xf16>
  vector.store %86, %alloc[%71, %32] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  %87 = vector.extract %6#0[1] : vector<2x2xf16>
  vector.store %87, %alloc[%73, %32] : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  gpu.barrier
  %88 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32)>()[%0, %1, %2]
  %89 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 32) * 256)>()[%0]
  %90 = vector.transfer_read %alloc[%88, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %91 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32)>()[%0, %1, %2, %workgroup_id_y]
  %92 = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 256 - (s0 floordiv 32) * 256)>()[%0, %workgroup_id_x]
  vector.transfer_write %90, %5[%91, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %93 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 8)>()[%0, %1, %2]
  %94 = vector.transfer_read %alloc[%93, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %95 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 8)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %94, %5[%95, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %96 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 16)>()[%0, %1, %2]
  %97 = vector.transfer_read %alloc[%96, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %98 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 16)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %97, %5[%98, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %99 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 24)>()[%0, %1, %2]
  %100 = vector.transfer_read %alloc[%99, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %101 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 24)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %100, %5[%101, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %102 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 32)>()[%0, %1, %2]
  %103 = vector.transfer_read %alloc[%102, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %104 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 32)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %103, %5[%104, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %105 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 40)>()[%0, %1, %2]
  %106 = vector.transfer_read %alloc[%105, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %107 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 40)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %106, %5[%107, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %108 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 48)>()[%0, %1, %2]
  %109 = vector.transfer_read %alloc[%108, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %110 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 48)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %109, %5[%110, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %111 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 56)>()[%0, %1, %2]
  %112 = vector.transfer_read %alloc[%111, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %113 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 56)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %112, %5[%113, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %114 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 64)>()[%0, %1, %2]
  %115 = vector.transfer_read %alloc[%114, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %116 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 64)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %115, %5[%116, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %117 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 72)>()[%0, %1, %2]
  %118 = vector.transfer_read %alloc[%117, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %119 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 72)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %118, %5[%119, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %120 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 80)>()[%0, %1, %2]
  %121 = vector.transfer_read %alloc[%120, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %122 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 80)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %121, %5[%122, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %123 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 88)>()[%0, %1, %2]
  %124 = vector.transfer_read %alloc[%123, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %125 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 88)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %124, %5[%125, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %126 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 96)>()[%0, %1, %2]
  %127 = vector.transfer_read %alloc[%126, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %128 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 96)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %127, %5[%128, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %129 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 104)>()[%0, %1, %2]
  %130 = vector.transfer_read %alloc[%129, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %131 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 104)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %130, %5[%131, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %132 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 112)>()[%0, %1, %2]
  %133 = vector.transfer_read %alloc[%132, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %134 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 112)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %133, %5[%134, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  %135 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 32 + 120)>()[%0, %1, %2]
  %136 = vector.transfer_read %alloc[%135, %89], %cst_0 {in_bounds = [true]} : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %137 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 4 + s2 * 8 + s3 * 128 + s0 floordiv 32 + 120)>()[%0, %1, %2, %workgroup_id_y]
  vector.transfer_write %136, %5[%137, %92] {in_bounds = [true]} : vector<8xf16>, memref<512x1280xf16>
  gpu.barrier
  return
}

//  CHECK-NV-LABEL: func.func @nvidia_tenscore_schedule
//  CHECK-NV-COUNT-6:  nvgpu.device_async_copy
//          CHECK-NV:  nvgpu.device_async_create_group
//  CHECK-NV-COUNT-6:  nvgpu.device_async_copy
//          CHECK-NV:  nvgpu.device_async_create_group
//          CHECK-NV:  nvgpu.device_async_wait %{{.*}} {numGroups = 1 : i32}
//          CHECK-NV:  gpu.barrier
//  CHECK-NV-COUNT-8:  nvgpu.ldmatrix
//          CHECK-NV:  scf.for
//  CHECK-NV-COUNT-8:    nvgpu.ldmatrix
// CHECK-NV-COUNT-64:    nvgpu.mma.sync
//  CHECK-NV-COUNT-6:    nvgpu.device_async_copy
//          CHECK-NV:    nvgpu.device_async_create_group
//          CHECK-NV:    nvgpu.device_async_wait %{{.*}} {numGroups = 1 : i32}
//          CHECK-NV:    gpu.barrier
//  CHECK-NV-COUNT-8:    nvgpu.ldmatrix
//          CHECK-NV:  }
//          CHECK-NV:  vector.store