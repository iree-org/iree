// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-lowering,canonicalize,cse))" --split-input-file %s | FileCheck %s

// -----// IR Dump After FoldTensorExtractOpPass (iree-codegen-fold-tensor-extract-op) ('func.func' operation: @main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32) //----- //
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none", waves_per_eu = 2 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#device_target_hip = #hal.device.target<"hip", [#executable_target_rocm_hsaco_fb]> : !hal.device
// module {
//   util.global private @__device_0 = #device_target_hip
//   hal.executable private @main$async_dispatch_2 {
//     hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
//       hal.executable.export public @main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32 ordinal(0) layout(#pipeline_layout) attributes {subgroup_size = 64 : index, workgroup_size = [256 : index, 1 : index, 1 : index]} {
//       ^bb0(%arg0: !hal.device):
//         %c4 = arith.constant 4 : index
//         %c128 = arith.constant 128 : index
//         %c1 = arith.constant 1 : index
//         hal.return %c4, %c128, %c1 : index, index, index
//       }
//       builtin.module {
        func.func @main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32() {
          %c32 = arith.constant 32 : index
          %c128 = arith.constant 128 : index
          %c48 = arith.constant 48 : index
          %c768 = arith.constant 768 : index
          %c512 = arith.constant 512 : index
          %c256 = arith.constant 256 : index
          %c2 = arith.constant 2 : index
          %c4 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %c64 = arith.constant 64 : index
          %cst = arith.constant dense<0.000000e+00> : vector<2x8x4x1xf32>
          %c8 = arith.constant 8 : index
          %cst_0 = arith.constant dense<0.000000e+00> : vector<1x2x8x4x1xf32>
          %cst_1 = arith.constant 0.000000e+00 : f16
          %c1 = arith.constant 1 : index
          %c131072 = arith.constant 131072 : index
          %c0 = arith.constant 0 : index
          %c671872 = arith.constant 671872 : index
          %c17449088 = arith.constant 17449088 : index
          %alloc = memref.alloc() : memref<512x20xf16, #gpu.address_space<workgroup>>
          %alloc_2 = memref.alloc() : memref<1x32x20xf16, #gpu.address_space<workgroup>>
          %alloca = memref.alloca() : memref<1x2x4x8x1xf32, #gpu.address_space<private>>
          %thread_id_x = gpu.thread_id  x upper_bound 256
          %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c131072) flags("ReadOnly|Indirect") : memref<130x130x16xf16, strided<[2080, 16, 1], offset: ?>, #gpu.address_space<global>>
          %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<130x130x16xf16, strided<[2080, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
          memref.assume_alignment %1, 64 : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
          %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<512x144xf16, #gpu.address_space<global>>
          %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<512x144xf16, #gpu.address_space<global>> to memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>
          memref.assume_alignment %3, 64 : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>
          %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x16xf16, #gpu.address_space<global>>
          %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<32x16xf16, #gpu.address_space<global>> to memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>
          memref.assume_alignment %5, 64 : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>
          %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c671872) flags(Indirect) : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1], offset: ?>, #gpu.address_space<global>>
          %7 = amdgpu.fat_raw_buffer_cast %6 resetOffset : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
          memref.assume_alignment %7, 64 : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
          %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c17449088) flags(Indirect) : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1], offset: ?>, #gpu.address_space<global>>
          %9 = amdgpu.fat_raw_buffer_cast %8 resetOffset : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
          memref.assume_alignment %9, 64 : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
          %subview = memref.subview %alloc_2[0, 0, 0] [1, 32, 16] [1, 1, 1] : memref<1x32x20xf16, #gpu.address_space<workgroup>> to memref<1x32x16xf16, strided<[640, 20, 1]>, #gpu.address_space<workgroup>>
          %subview_3 = memref.subview %alloc[0, 0] [512, 16] [1, 1] : memref<512x20xf16, #gpu.address_space<workgroup>> to memref<512x16xf16, strided<[20, 1]>, #gpu.address_space<workgroup>>
          %10 = arith.floordivsi %thread_id_x, %c64 : index
          %11 = arith.muli %10, %c8 overflow<nsw> : index
          %12 = gpu.lane_id upper_bound 64
          %13 = arith.remsi %12, %c64 : index
          %14 = arith.cmpi slt, %13, %c0 : index
          %15 = arith.addi %13, %c64 : index
          %16 = arith.select %14, %15, %13 : index
          %17 = arith.divsi %16, %c16 : index
          %18 = arith.remsi %12, %c16 : index
          %19 = arith.cmpi slt, %18, %c0 : index
          %20 = arith.addi %18, %c16 : index
          %21 = arith.select %19, %20, %18 : index
          %22 = arith.muli %17, %c4 : index
          %23 = arith.muli %10, %c64 overflow<nsw> : index
          %24 = arith.addi %12, %23 : index
          %25 = arith.floordivsi %24, %c8 : index
          %26 = arith.remsi %24, %c8 : index
          %27 = arith.cmpi slt, %26, %c0 : index
          %28 = arith.addi %26, %c8 : index
          %29 = arith.select %27, %28, %26 : index
          %30 = arith.floordivsi %24, %c2 : index
          %31 = arith.remsi %24, %c2 : index
          %32 = arith.cmpi slt, %31, %c0 : index
          %33 = arith.addi %31, %c2 : index
          %34 = arith.select %32, %33, %31 : index
          %35 = arith.addi %24, %c256 : index
          %36 = arith.floordivsi %35, %c2 : index
          %37 = arith.remsi %35, %c2 : index
          %38 = arith.cmpi slt, %37, %c0 : index
          %39 = arith.addi %37, %c2 : index
          %40 = arith.select %38, %39, %37 : index
          %41 = arith.addi %24, %c512 : index
          %42 = arith.floordivsi %41, %c2 : index
          %43 = arith.remsi %41, %c2 : index
          %44 = arith.cmpi slt, %43, %c0 : index
          %45 = arith.addi %43, %c2 : index
          %46 = arith.select %44, %45, %43 : index
          %47 = arith.addi %24, %c768 : index
          %48 = arith.floordivsi %47, %c2 : index
          %49 = arith.remsi %47, %c2 : index
          %50 = arith.cmpi slt, %49, %c0 : index
          %51 = arith.addi %49, %c2 : index
          %52 = arith.select %50, %51, %49 : index
          %expand_shape = memref.expand_shape %subview_3 [[0, 1], [2, 3]] output_shape [32, 16, 1, 16] : memref<512x16xf16, strided<[20, 1]>, #gpu.address_space<workgroup>> into memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>
          %expand_shape_4 = memref.expand_shape %subview [[0], [1, 2], [3, 4]] output_shape [1, 2, 16, 1, 16] : memref<1x32x16xf16, strided<[640, 20, 1]>, #gpu.address_space<workgroup>> into memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>
          %53 = arith.muli %29, %c2 overflow<nsw> : index
          %54 = arith.floordivsi %53, %c48 : index
          %55 = arith.remsi %53, %c48 : index
          %56 = arith.cmpi slt, %55, %c0 : index
          %57 = arith.addi %55, %c48 : index
          %58 = arith.select %56, %57, %55 : index
          %59 = arith.divsi %58, %c16 : index
          %60 = arith.remsi %53, %c16 : index
          %61 = arith.cmpi slt, %60, %c0 : index
          %62 = arith.addi %60, %c16 : index
          %63 = arith.select %61, %62, %60 : index
          %64 = arith.muli %34, %c8 overflow<nsw> : index
          %65 = arith.muli %40, %c8 overflow<nsw> : index
          %66 = arith.muli %46, %c8 overflow<nsw> : index
          %67 = arith.muli %52, %c8 overflow<nsw> : index
          %subview_5 = memref.subview %alloca[0, 0, 0, 0, 0] [1, 2, 4, 8, 1] [1, 1, 1, 1, 1] : memref<1x2x4x8x1xf32, #gpu.address_space<private>> to memref<1x2x4x8xf32, strided<[64, 32, 8, 1]>, #gpu.address_space<private>>
          %workgroup_id_x = hal.interface.workgroup.id[0] : index
          %workgroup_id_y = hal.interface.workgroup.id[1] : index
          gpu.barrier
          %68 = arith.muli %workgroup_id_y, %c128 overflow<nsw> : index
          %69 = arith.addi %25, %68 : index
          %70 = arith.muli %workgroup_id_x, %c32 overflow<nsw> : index
          %71 = arith.addi %69, %70 : index
          %72 = arith.floordivsi %71, %c128 : index
          %73 = arith.remsi %71, %c128 : index
          %74 = arith.cmpi slt, %73, %c0 : index
          %75 = arith.addi %73, %c128 : index
          %76 = arith.select %74, %75, %73 : index
          %77 = arith.addi %54, %72 : index
          %78 = arith.addi %59, %76 : index
          %79 = vector.transfer_read %1[%77, %78, %63], %cst_1 {in_bounds = [true]} : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf16>
          %80 = vector.transfer_read %3[%30, %64], %cst_1 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
          %81 = vector.transfer_read %3[%36, %65], %cst_1 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
          %82 = vector.transfer_read %3[%42, %66], %cst_1 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
          %83 = vector.transfer_read %3[%48, %67], %cst_1 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
          vector.transfer_write %79, %alloc_2[%c0, %25, %53] {in_bounds = [true]} : vector<2xf16>, memref<1x32x20xf16, #gpu.address_space<workgroup>>
          vector.transfer_write %80, %alloc[%30, %64] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
          vector.transfer_write %81, %alloc[%36, %65] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
          vector.transfer_write %82, %alloc[%42, %66] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
          vector.transfer_write %83, %alloc[%48, %67] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
          %84 = scf.for %arg0 = %c0 to %c8 step %c1 iter_args(%arg1 = %cst_0) -> (vector<1x2x8x4x1xf32>) {
            %200 = arith.addi %arg0, %c1 : index
            %201 = arith.muli %200, %c16 overflow<nsw> : index
            %202 = arith.addi %201, %53 : index
            %203 = arith.floordivsi %202, %c48 : index
            %204 = arith.remsi %202, %c48 : index
            %205 = arith.cmpi slt, %204, %c0 : index
            %206 = arith.addi %204, %c48 : index
            %207 = arith.select %205, %206, %204 : index
            %208 = arith.divsi %207, %c16 : index
            %209 = arith.remsi %202, %c16 : index
            %210 = arith.cmpi slt, %209, %c0 : index
            %211 = arith.addi %209, %c16 : index
            %212 = arith.select %210, %211, %209 : index
            %213 = arith.addi %203, %72 : index
            %214 = arith.addi %208, %76 : index
            %215 = vector.transfer_read %1[%213, %214, %212], %cst_1 {in_bounds = [true]} : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf16>
            %216 = arith.addi %201, %64 : index
            %217 = vector.transfer_read %3[%30, %216], %cst_1 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %218 = arith.addi %201, %65 : index
            %219 = vector.transfer_read %3[%36, %218], %cst_1 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %220 = arith.addi %201, %66 : index
            %221 = vector.transfer_read %3[%42, %220], %cst_1 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            %222 = arith.addi %201, %67 : index
            %223 = vector.transfer_read %3[%48, %222], %cst_1 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
            gpu.barrier
            %224 = vector.transfer_read %expand_shape_4[%c0, %c0, %21, %c0, %22], %cst_1 {in_bounds = [true, true, true, true]} : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<2x1x1x4xf16>
            %225 = vector.transfer_read %expand_shape[%11, %21, %c0, %22], %cst_1 {in_bounds = [true, true, true, true]} : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<8x1x1x4xf16>
            %226 = vector.transpose %225, [0, 2, 1, 3] : vector<8x1x1x4xf16> to vector<8x1x1x4xf16>
            %227 = vector.extract %224[0, 0] : vector<1x4xf16> from vector<2x1x1x4xf16>
            %228 = vector.extract %226[0, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
            %229 = vector.extract %arg1[0, 0, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %230 = vector.shape_cast %227 : vector<1x4xf16> to vector<4xf16>
            %231 = vector.shape_cast %228 : vector<1x4xf16> to vector<4xf16>
            %232 = vector.shape_cast %229 : vector<4x1xf32> to vector<4xf32>
            %233 = amdgpu.mfma %230 * %231 + %232 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %234 = vector.shape_cast %233 : vector<4xf32> to vector<4x1xf32>
            %235 = vector.extract %226[1, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
            %236 = vector.extract %arg1[0, 0, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %237 = vector.shape_cast %235 : vector<1x4xf16> to vector<4xf16>
            %238 = vector.shape_cast %236 : vector<4x1xf32> to vector<4xf32>
            %239 = amdgpu.mfma %230 * %237 + %238 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %240 = vector.shape_cast %239 : vector<4xf32> to vector<4x1xf32>
            %241 = vector.extract %226[2, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
            %242 = vector.extract %arg1[0, 0, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %243 = vector.shape_cast %241 : vector<1x4xf16> to vector<4xf16>
            %244 = vector.shape_cast %242 : vector<4x1xf32> to vector<4xf32>
            %245 = amdgpu.mfma %230 * %243 + %244 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %246 = vector.shape_cast %245 : vector<4xf32> to vector<4x1xf32>
            %247 = vector.extract %226[3, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
            %248 = vector.extract %arg1[0, 0, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %249 = vector.shape_cast %247 : vector<1x4xf16> to vector<4xf16>
            %250 = vector.shape_cast %248 : vector<4x1xf32> to vector<4xf32>
            %251 = amdgpu.mfma %230 * %249 + %250 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %252 = vector.shape_cast %251 : vector<4xf32> to vector<4x1xf32>
            %253 = vector.extract %226[4, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
            %254 = vector.extract %arg1[0, 0, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %255 = vector.shape_cast %253 : vector<1x4xf16> to vector<4xf16>
            %256 = vector.shape_cast %254 : vector<4x1xf32> to vector<4xf32>
            %257 = amdgpu.mfma %230 * %255 + %256 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %258 = vector.shape_cast %257 : vector<4xf32> to vector<4x1xf32>
            %259 = vector.extract %226[5, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
            %260 = vector.extract %arg1[0, 0, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %261 = vector.shape_cast %259 : vector<1x4xf16> to vector<4xf16>
            %262 = vector.shape_cast %260 : vector<4x1xf32> to vector<4xf32>
            %263 = amdgpu.mfma %230 * %261 + %262 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %264 = vector.shape_cast %263 : vector<4xf32> to vector<4x1xf32>
            %265 = vector.extract %226[6, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
            %266 = vector.extract %arg1[0, 0, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %267 = vector.shape_cast %265 : vector<1x4xf16> to vector<4xf16>
            %268 = vector.shape_cast %266 : vector<4x1xf32> to vector<4xf32>
            %269 = amdgpu.mfma %230 * %267 + %268 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %270 = vector.shape_cast %269 : vector<4xf32> to vector<4x1xf32>
            %271 = vector.extract %226[7, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
            %272 = vector.extract %arg1[0, 0, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %273 = vector.shape_cast %271 : vector<1x4xf16> to vector<4xf16>
            %274 = vector.shape_cast %272 : vector<4x1xf32> to vector<4xf32>
            %275 = amdgpu.mfma %230 * %273 + %274 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %276 = vector.shape_cast %275 : vector<4xf32> to vector<4x1xf32>
            %277 = vector.extract %224[1, 0] : vector<1x4xf16> from vector<2x1x1x4xf16>
            %278 = vector.extract %arg1[0, 1, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %279 = vector.shape_cast %277 : vector<1x4xf16> to vector<4xf16>
            %280 = vector.shape_cast %278 : vector<4x1xf32> to vector<4xf32>
            %281 = amdgpu.mfma %279 * %231 + %280 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %282 = vector.shape_cast %281 : vector<4xf32> to vector<4x1xf32>
            %283 = vector.extract %arg1[0, 1, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %284 = vector.shape_cast %283 : vector<4x1xf32> to vector<4xf32>
            %285 = amdgpu.mfma %279 * %237 + %284 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %286 = vector.shape_cast %285 : vector<4xf32> to vector<4x1xf32>
            %287 = vector.extract %arg1[0, 1, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %288 = vector.shape_cast %287 : vector<4x1xf32> to vector<4xf32>
            %289 = amdgpu.mfma %279 * %243 + %288 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %290 = vector.shape_cast %289 : vector<4xf32> to vector<4x1xf32>
            %291 = vector.extract %arg1[0, 1, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %292 = vector.shape_cast %291 : vector<4x1xf32> to vector<4xf32>
            %293 = amdgpu.mfma %279 * %249 + %292 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %294 = vector.shape_cast %293 : vector<4xf32> to vector<4x1xf32>
            %295 = vector.extract %arg1[0, 1, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %296 = vector.shape_cast %295 : vector<4x1xf32> to vector<4xf32>
            %297 = amdgpu.mfma %279 * %255 + %296 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %298 = vector.shape_cast %297 : vector<4xf32> to vector<4x1xf32>
            %299 = vector.extract %arg1[0, 1, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %300 = vector.shape_cast %299 : vector<4x1xf32> to vector<4xf32>
            %301 = amdgpu.mfma %279 * %261 + %300 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %302 = vector.shape_cast %301 : vector<4xf32> to vector<4x1xf32>
            %303 = vector.extract %arg1[0, 1, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %304 = vector.shape_cast %303 : vector<4x1xf32> to vector<4xf32>
            %305 = amdgpu.mfma %279 * %267 + %304 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %306 = vector.shape_cast %305 : vector<4xf32> to vector<4x1xf32>
            %307 = vector.extract %arg1[0, 1, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
            %308 = vector.shape_cast %307 : vector<4x1xf32> to vector<4xf32>
            %309 = amdgpu.mfma %279 * %273 + %308 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %310 = vector.shape_cast %309 : vector<4xf32> to vector<4x1xf32>
            %311 = vector.insert_strided_slice %234, %cst {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %312 = vector.insert_strided_slice %240, %311 {offsets = [0, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %313 = vector.insert_strided_slice %246, %312 {offsets = [0, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %314 = vector.insert_strided_slice %252, %313 {offsets = [0, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %315 = vector.insert_strided_slice %258, %314 {offsets = [0, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %316 = vector.insert_strided_slice %264, %315 {offsets = [0, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %317 = vector.insert_strided_slice %270, %316 {offsets = [0, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %318 = vector.insert_strided_slice %276, %317 {offsets = [0, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %319 = vector.insert_strided_slice %282, %318 {offsets = [1, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %320 = vector.insert_strided_slice %286, %319 {offsets = [1, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %321 = vector.insert_strided_slice %290, %320 {offsets = [1, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %322 = vector.insert_strided_slice %294, %321 {offsets = [1, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %323 = vector.insert_strided_slice %298, %322 {offsets = [1, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %324 = vector.insert_strided_slice %302, %323 {offsets = [1, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %325 = vector.insert_strided_slice %306, %324 {offsets = [1, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %326 = vector.insert_strided_slice %310, %325 {offsets = [1, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
            %327 = vector.broadcast %326 : vector<2x8x4x1xf32> to vector<1x2x8x4x1xf32>
            gpu.barrier
            vector.transfer_write %215, %alloc_2[%c0, %25, %53] {in_bounds = [true]} : vector<2xf16>, memref<1x32x20xf16, #gpu.address_space<workgroup>>
            vector.transfer_write %217, %alloc[%30, %64] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
            vector.transfer_write %219, %alloc[%36, %65] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
            vector.transfer_write %221, %alloc[%42, %66] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
            vector.transfer_write %223, %alloc[%48, %67] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
            scf.yield %327 : vector<1x2x8x4x1xf32>
          }
          gpu.barrier
          %85 = vector.transfer_read %expand_shape_4[%c0, %c0, %21, %c0, %22], %cst_1 {in_bounds = [true, true, true, true]} : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<2x1x1x4xf16>
          %86 = vector.transfer_read %expand_shape[%11, %21, %c0, %22], %cst_1 {in_bounds = [true, true, true, true]} : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<8x1x1x4xf16>
          %87 = vector.transpose %86, [0, 2, 1, 3] : vector<8x1x1x4xf16> to vector<8x1x1x4xf16>
          %88 = vector.extract %85[0, 0] : vector<1x4xf16> from vector<2x1x1x4xf16>
          %89 = vector.extract %87[0, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
          %90 = vector.extract %84[0, 0, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %91 = vector.shape_cast %88 : vector<1x4xf16> to vector<4xf16>
          %92 = vector.shape_cast %89 : vector<1x4xf16> to vector<4xf16>
          %93 = vector.shape_cast %90 : vector<4x1xf32> to vector<4xf32>
          %94 = amdgpu.mfma %91 * %92 + %93 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %95 = vector.shape_cast %94 : vector<4xf32> to vector<4x1xf32>
          %96 = vector.extract %87[1, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
          %97 = vector.extract %84[0, 0, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %98 = vector.shape_cast %96 : vector<1x4xf16> to vector<4xf16>
          %99 = vector.shape_cast %97 : vector<4x1xf32> to vector<4xf32>
          %100 = amdgpu.mfma %91 * %98 + %99 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %101 = vector.shape_cast %100 : vector<4xf32> to vector<4x1xf32>
          %102 = vector.extract %87[2, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
          %103 = vector.extract %84[0, 0, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %104 = vector.shape_cast %102 : vector<1x4xf16> to vector<4xf16>
          %105 = vector.shape_cast %103 : vector<4x1xf32> to vector<4xf32>
          %106 = amdgpu.mfma %91 * %104 + %105 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %107 = vector.shape_cast %106 : vector<4xf32> to vector<4x1xf32>
          %108 = vector.extract %87[3, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
          %109 = vector.extract %84[0, 0, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %110 = vector.shape_cast %108 : vector<1x4xf16> to vector<4xf16>
          %111 = vector.shape_cast %109 : vector<4x1xf32> to vector<4xf32>
          %112 = amdgpu.mfma %91 * %110 + %111 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %113 = vector.shape_cast %112 : vector<4xf32> to vector<4x1xf32>
          %114 = vector.extract %87[4, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
          %115 = vector.extract %84[0, 0, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %116 = vector.shape_cast %114 : vector<1x4xf16> to vector<4xf16>
          %117 = vector.shape_cast %115 : vector<4x1xf32> to vector<4xf32>
          %118 = amdgpu.mfma %91 * %116 + %117 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %119 = vector.shape_cast %118 : vector<4xf32> to vector<4x1xf32>
          %120 = vector.extract %87[5, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
          %121 = vector.extract %84[0, 0, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %122 = vector.shape_cast %120 : vector<1x4xf16> to vector<4xf16>
          %123 = vector.shape_cast %121 : vector<4x1xf32> to vector<4xf32>
          %124 = amdgpu.mfma %91 * %122 + %123 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %125 = vector.shape_cast %124 : vector<4xf32> to vector<4x1xf32>
          %126 = vector.extract %87[6, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
          %127 = vector.extract %84[0, 0, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %128 = vector.shape_cast %126 : vector<1x4xf16> to vector<4xf16>
          %129 = vector.shape_cast %127 : vector<4x1xf32> to vector<4xf32>
          %130 = amdgpu.mfma %91 * %128 + %129 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %131 = vector.shape_cast %130 : vector<4xf32> to vector<4x1xf32>
          %132 = vector.extract %87[7, 0] : vector<1x4xf16> from vector<8x1x1x4xf16>
          %133 = vector.extract %84[0, 0, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %134 = vector.shape_cast %132 : vector<1x4xf16> to vector<4xf16>
          %135 = vector.shape_cast %133 : vector<4x1xf32> to vector<4xf32>
          %136 = amdgpu.mfma %91 * %134 + %135 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %137 = vector.shape_cast %136 : vector<4xf32> to vector<4x1xf32>
          %138 = vector.extract %85[1, 0] : vector<1x4xf16> from vector<2x1x1x4xf16>
          %139 = vector.extract %84[0, 1, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %140 = vector.shape_cast %138 : vector<1x4xf16> to vector<4xf16>
          %141 = vector.shape_cast %139 : vector<4x1xf32> to vector<4xf32>
          %142 = amdgpu.mfma %140 * %92 + %141 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %143 = vector.shape_cast %142 : vector<4xf32> to vector<4x1xf32>
          %144 = vector.extract %84[0, 1, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %145 = vector.shape_cast %144 : vector<4x1xf32> to vector<4xf32>
          %146 = amdgpu.mfma %140 * %98 + %145 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %147 = vector.shape_cast %146 : vector<4xf32> to vector<4x1xf32>
          %148 = vector.extract %84[0, 1, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %149 = vector.shape_cast %148 : vector<4x1xf32> to vector<4xf32>
          %150 = amdgpu.mfma %140 * %104 + %149 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %151 = vector.shape_cast %150 : vector<4xf32> to vector<4x1xf32>
          %152 = vector.extract %84[0, 1, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %153 = vector.shape_cast %152 : vector<4x1xf32> to vector<4xf32>
          %154 = amdgpu.mfma %140 * %110 + %153 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %155 = vector.shape_cast %154 : vector<4xf32> to vector<4x1xf32>
          %156 = vector.extract %84[0, 1, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %157 = vector.shape_cast %156 : vector<4x1xf32> to vector<4xf32>
          %158 = amdgpu.mfma %140 * %116 + %157 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %159 = vector.shape_cast %158 : vector<4xf32> to vector<4x1xf32>
          %160 = vector.extract %84[0, 1, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %161 = vector.shape_cast %160 : vector<4x1xf32> to vector<4xf32>
          %162 = amdgpu.mfma %140 * %122 + %161 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %163 = vector.shape_cast %162 : vector<4xf32> to vector<4x1xf32>
          %164 = vector.extract %84[0, 1, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %165 = vector.shape_cast %164 : vector<4x1xf32> to vector<4xf32>
          %166 = amdgpu.mfma %140 * %128 + %165 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %167 = vector.shape_cast %166 : vector<4xf32> to vector<4x1xf32>
          %168 = vector.extract %84[0, 1, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
          %169 = vector.shape_cast %168 : vector<4x1xf32> to vector<4xf32>
          %170 = amdgpu.mfma %140 * %134 + %169 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %171 = vector.shape_cast %170 : vector<4xf32> to vector<4x1xf32>
          %172 = vector.insert_strided_slice %95, %cst {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %173 = vector.insert_strided_slice %101, %172 {offsets = [0, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %174 = vector.insert_strided_slice %107, %173 {offsets = [0, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %175 = vector.insert_strided_slice %113, %174 {offsets = [0, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %176 = vector.insert_strided_slice %119, %175 {offsets = [0, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %177 = vector.insert_strided_slice %125, %176 {offsets = [0, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %178 = vector.insert_strided_slice %131, %177 {offsets = [0, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %179 = vector.insert_strided_slice %137, %178 {offsets = [0, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %180 = vector.insert_strided_slice %143, %179 {offsets = [1, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %181 = vector.insert_strided_slice %147, %180 {offsets = [1, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %182 = vector.insert_strided_slice %151, %181 {offsets = [1, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %183 = vector.insert_strided_slice %155, %182 {offsets = [1, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %184 = vector.insert_strided_slice %159, %183 {offsets = [1, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %185 = vector.insert_strided_slice %163, %184 {offsets = [1, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %186 = vector.insert_strided_slice %167, %185 {offsets = [1, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %187 = vector.insert_strided_slice %171, %186 {offsets = [1, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
          %188 = vector.broadcast %187 : vector<2x8x4x1xf32> to vector<1x2x8x4x1xf32>
          %189 = vector.transpose %188, [0, 1, 3, 2, 4] : vector<1x2x8x4x1xf32> to vector<1x2x4x8x1xf32>
          %190 = vector.extract %189[0] : vector<2x4x8x1xf32> from vector<1x2x4x8x1xf32>
          %191 = vector.shape_cast %190 : vector<2x4x8x1xf32> to vector<2x4x8xf32>
          vector.transfer_write %191, %subview_5[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<2x4x8xf32>, memref<1x2x4x8xf32, strided<[64, 32, 8, 1]>, #gpu.address_space<private>>
          %192 = vector.transfer_read %5[%11, %21], %cst_1 {in_bounds = [true, true]} : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8x1xf16>
          %193 = arith.extf %192 : vector<8x1xf16> to vector<8x1xf32>
          %194 = vector.broadcast %193 : vector<8x1xf32> to vector<2x4x8x1xf32>
          %195 = arith.addf %190, %194 : vector<2x4x8x1xf32>
          %196 = arith.truncf %195 : vector<2x4x8x1xf32> to vector<2x4x8x1xf16>
          %197 = vector.broadcast %196 : vector<2x4x8x1xf16> to vector<1x2x4x8x1xf16>
          %198 = arith.muli %workgroup_id_x, %c2 overflow<nsw> : index
          vector.transfer_write %196, %7[%workgroup_id_y, %198, %22, %11, %21] {in_bounds = [true, true, true, true]} : vector<2x4x8x1xf16>, memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
          %199 = vector.transpose %197, [3, 4, 0, 1, 2] : vector<1x2x4x8x1xf16> to vector<8x1x1x2x4xf16>
          vector.transfer_write %199, %9[%11, %21, %workgroup_id_y, %198, %22] {in_bounds = [true, true, true, true, true]} : vector<8x1x1x2x4xf16>, memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
          gpu.barrier
          memref.dealloc %alloc_2 : memref<1x32x20xf16, #gpu.address_space<workgroup>>
          memref.dealloc %alloc : memref<512x20xf16, #gpu.address_space<workgroup>>
          return
        }
//       }
//     }
//   }
//   util.global private @main$async_dispatch_2_rocm_hsaco_fb_main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32_buffer : !hal.buffer
//   util.initializer {
//     %c8155990016 = arith.constant 8155990016 : index
//     %device, %queue_affinity = hal.device.resolve on(<@__device_0>) : !hal.device, i64
//     %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
//     %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%queue_affinity) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c8155990016}
//     util.global.store %buffer, @main$async_dispatch_2_rocm_hsaco_fb_main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32_buffer : !hal.buffer
//     util.return
//   }
//   util.func public @main$async_dispatch_2_rocm_hsaco_fb_main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
//     %c-1_i32 = arith.constant -1 : i32
//     %0 = util.null : !hal.fence
//     %c1 = arith.constant 1 : index
//     %c5469536256 = arith.constant 5469536256 : index
//     %c2783082496 = arith.constant 2783082496 : index
//     %c1024 = arith.constant 1024 : index
//     %c2783081472 = arith.constant 2783081472 : index
//     %c96627712 = arith.constant 96627712 : index
//     %c2686453760 = arith.constant 2686453760 : index
//     %c0 = arith.constant 0 : index
//     %main$async_dispatch_2_rocm_hsaco_fb_main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32_buffer = util.global.load immutable @main$async_dispatch_2_rocm_hsaco_fb_main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32_buffer : !hal.buffer
//     %1 = arith.index_cast %arg0 : i32 to index
//     %device, %queue_affinity = hal.device.resolve on(<@__device_0>) : !hal.device, i64
//     %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) affinity(%queue_affinity) : !hal.command_buffer
//     %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device : !hal.device) target(@main$async_dispatch_2::@rocm_hsaco_fb::@main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32) : index, index, index
//     %exe = hal.executable.lookup device(%device : !hal.device) executable(@main$async_dispatch_2) : !hal.executable
//     %ordinal = hal.executable.export.ordinal target(@main$async_dispatch_2::@rocm_hsaco_fb::@main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32) : index
//     cf.br ^bb1(%c0 : index)
//   ^bb1(%2: index):  // 2 preds: ^bb0, ^bb2
//     %3 = arith.cmpi slt, %2, %1 : index
//     cf.cond_br %3, ^bb2, ^bb3
//   ^bb2:  // pred: ^bb1
//     hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z]) bindings([
//       (%main$async_dispatch_2_rocm_hsaco_fb_main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32_buffer : !hal.buffer)[%c0, %c2686453760], 
//       (%main$async_dispatch_2_rocm_hsaco_fb_main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32_buffer : !hal.buffer)[%c2686453760, %c96627712], 
//       (%main$async_dispatch_2_rocm_hsaco_fb_main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32_buffer : !hal.buffer)[%c2783081472, %c1024], 
//       (%main$async_dispatch_2_rocm_hsaco_fb_main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32_buffer : !hal.buffer)[%c2783082496, %c2686453760], 
//       (%main$async_dispatch_2_rocm_hsaco_fb_main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32_buffer : !hal.buffer)[%c5469536256, %c2686453760]
//     ]) flags("None")
//     hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
//     %4 = arith.addi %2, %c1 : index
//     cf.br ^bb1(%4 : index)
//   ^bb3:  // pred: ^bb1
//     hal.command_buffer.finalize<%cmd : !hal.command_buffer>
//     %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
//     hal.device.queue.execute<%device : !hal.device> affinity(%queue_affinity) wait(%0) signal(%fence) commands(%cmd) flags("None")
//     %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
//     util.status.check_ok %status, "failed to wait on timepoint"
//     util.return
//   }
// }
// 
// 
