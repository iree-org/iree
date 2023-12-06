// RUN: iree-opt  %s --iree-transform-dialect-interpreter --transform-dialect-drop-schedule | FileCheck %s

builtin.module attributes { transform.with_named_sequence } {
  func.func @matmul() {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<2x2x16xf32>
    %c4096 = arith.constant 4096 : index
    %c32 = arith.constant 32 : index
    %cst_1 = arith.constant dense<0.000000e+00> : vector<2x2x8xf16>
    %0 = gpu.thread_id  x
    %1 = gpu.thread_id  y
    %2 = gpu.thread_id  z  
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x32xf16, #gpu.address_space<workgroup>>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x32xf16, #gpu.address_space<workgroup>>
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>
    memref.assume_alignment %3, 64 : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>
    memref.assume_alignment %4, 64 : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    memref.assume_alignment %5, 64 : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %6 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 256 + s2 * 64 + (s1 floordiv 64) * 64 - ((s2 + s1 floordiv 64) floordiv 4) * 256)>()[%workgroup_id_x, %0, %1]
    %7 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 128 + ((s2 + s1 floordiv 64) floordiv 4) * 64)>()[%workgroup_id_y, %0, %1]
    %8 = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + (s0 floordiv 64) * 64 - ((s1 + s0 floordiv 64) floordiv 4) * 256)>()[%0, %1]
    %9 = affine.apply affine_map<()[s0, s1] -> (((s1 + s0 floordiv 64) floordiv 4) * 64)>()[%0, %1]
    %10 = scf.for %arg0 = %c0 to %c4096 step %c32 iter_args(%arg1 = %cst_0) -> (vector<2x2x16xf32>) {
      %144 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 * 256 + s2 * 16 + s3 * 128 + s1 floordiv 4)>()[%workgroup_id_x, %0, %1, %2]
      %145 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8 - (s1 floordiv 4) * 32)>()[%arg0, %0]
      %146 = vector.transfer_read %3[%144, %145], %cst {in_bounds = [true, true]} : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
      %147 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 128 + s0 floordiv 4)>()[%0, %1, %2]
      %148 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
      vector.transfer_write %146, %alloc_2[%147, %148] {in_bounds = [true, true]} : vector<1x8xf16>, memref<256x32xf16, #gpu.address_space<workgroup>>
      %149 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 * 256 + s2 * 16 + s3 * 128 + s1 floordiv 4 + 128)>()[%workgroup_id_x, %0, %1, %2]
      %150 = vector.transfer_read %3[%149, %145], %cst {in_bounds = [true, true]} : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
      %151 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 128 + s0 floordiv 4 + 128)>()[%0, %1, %2]
      vector.transfer_write %150, %alloc_2[%151, %148] {in_bounds = [true, true]} : vector<1x8xf16>, memref<256x32xf16, #gpu.address_space<workgroup>>
      %152 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 * 128 + s2 * 16 + s3 * 128 + s1 floordiv 4)>()[%workgroup_id_y, %0, %1, %2]
      %153 = vector.transfer_read %4[%152, %145], %cst {in_bounds = [true, true]} : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
      vector.transfer_write %153, %alloc[%147, %148] {in_bounds = [true, true]} : vector<1x8xf16>, memref<128x32xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      gpu.barrier
      %154 = arith.remui %0, %c32 : index
      %155 = arith.divui %0, %c32 : index
      %156 = affine.apply affine_map<(d0) -> (d0 * 8)>(%155)
      %157 = arith.addi %154, %8 : index
      %158 = vector.load %alloc_2[%157, %156] : memref<256x32xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      %159 = vector.insert_strided_slice %158, %cst_1 {offsets = [0, 0, 0], strides = [1]} : vector<8xf16> into vector<2x2x8xf16>
      %160 = affine.apply affine_map<(d0) -> (d0 * 8 + 16)>(%155)
      %161 = vector.load %alloc_2[%157, %160] : memref<256x32xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      %162 = vector.insert_strided_slice %161, %159 {offsets = [0, 1, 0], strides = [1]} : vector<8xf16> into vector<2x2x8xf16>
      %163 = affine.apply affine_map<(d0) -> (d0 + 32)>(%154)
      %164 = arith.addi %163, %8 : index
      %165 = vector.load %alloc_2[%164, %156] : memref<256x32xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      %166 = vector.insert_strided_slice %165, %162 {offsets = [1, 0, 0], strides = [1]} : vector<8xf16> into vector<2x2x8xf16>
      %167 = vector.load %alloc_2[%164, %160] : memref<256x32xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      %168 = vector.insert_strided_slice %167, %166 {offsets = [1, 1, 0], strides = [1]} : vector<8xf16> into vector<2x2x8xf16>
      %169 = arith.addi %154, %9 : index
      %170 = vector.load %alloc[%169, %156] : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      %171 = vector.insert_strided_slice %170, %cst_1 {offsets = [0, 0, 0], strides = [1]} : vector<8xf16> into vector<2x2x8xf16>
      %172 = vector.load %alloc[%169, %160] : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      %173 = vector.insert_strided_slice %172, %171 {offsets = [0, 1, 0], strides = [1]} : vector<8xf16> into vector<2x2x8xf16>
      %174 = arith.addi %163, %9 : index
      %175 = vector.load %alloc[%174, %156] : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      %176 = vector.insert_strided_slice %175, %173 {offsets = [1, 0, 0], strides = [1]} : vector<8xf16> into vector<2x2x8xf16>
      %177 = vector.load %alloc[%174, %160] : memref<128x32xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      %178 = vector.insert_strided_slice %177, %176 {offsets = [1, 1, 0], strides = [1]} : vector<8xf16> into vector<2x2x8xf16>
      %179 = vector.extract_strided_slice %168 {offsets = [0, 0, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %180 = vector.extract_strided_slice %168 {offsets = [0, 0, 4], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %181 = vector.extract_strided_slice %168 {offsets = [0, 1, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %182 = vector.extract_strided_slice %168 {offsets = [0, 1, 4], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %183 = vector.extract_strided_slice %168 {offsets = [1, 0, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %184 = vector.extract_strided_slice %168 {offsets = [1, 0, 4], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %185 = vector.extract_strided_slice %168 {offsets = [1, 1, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %186 = vector.extract_strided_slice %168 {offsets = [1, 1, 4], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %187 = vector.extract_strided_slice %178 {offsets = [0, 0, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %188 = vector.extract_strided_slice %178 {offsets = [0, 0, 4], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %189 = vector.extract_strided_slice %178 {offsets = [0, 1, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %190 = vector.extract_strided_slice %178 {offsets = [0, 1, 4], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %191 = vector.extract_strided_slice %178 {offsets = [1, 0, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %192 = vector.extract_strided_slice %178 {offsets = [1, 0, 4], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %193 = vector.extract_strided_slice %178 {offsets = [1, 1, 0], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %194 = vector.extract_strided_slice %178 {offsets = [1, 1, 4], sizes = [1, 1, 4], strides = [1, 1, 1]} : vector<2x2x8xf16> to vector<1x1x4xf16>
      %195 = vector.extract %arg1[0, 0] : vector<16xf32> from vector<2x2x16xf32>
      %196 = vector.extract %179[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %197 = vector.extract %187[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %198 = amdgpu.mfma %196 * %197 + %195 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %199 = vector.extract %180[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %200 = vector.extract %188[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %201 = amdgpu.mfma %199 * %200 + %198 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %202 = vector.extract %181[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %203 = vector.extract %189[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %204 = amdgpu.mfma %202 * %203 + %201 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %205 = vector.extract %182[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %206 = vector.extract %190[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %207 = amdgpu.mfma %205 * %206 + %204 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %208 = vector.insert %207, %cst_0 [0, 0] : vector<16xf32> into vector<2x2x16xf32>
      %209 = vector.extract %arg1[0, 1] : vector<16xf32> from vector<2x2x16xf32>
      %210 = vector.extract %191[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %211 = amdgpu.mfma %196 * %210 + %209 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %212 = vector.extract %192[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %213 = amdgpu.mfma %199 * %212 + %211 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %214 = vector.extract %193[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %215 = amdgpu.mfma %202 * %214 + %213 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %216 = vector.extract %194[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %217 = amdgpu.mfma %205 * %216 + %215 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %218 = vector.insert %217, %208 [0, 1] : vector<16xf32> into vector<2x2x16xf32>
      %219 = vector.extract %arg1[1, 0] : vector<16xf32> from vector<2x2x16xf32>
      %220 = vector.extract %183[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %221 = amdgpu.mfma %220 * %197 + %219 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %222 = vector.extract %184[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %223 = amdgpu.mfma %222 * %200 + %221 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %224 = vector.extract %185[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %225 = amdgpu.mfma %224 * %203 + %223 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %226 = vector.extract %186[0, 0] : vector<4xf16> from vector<1x1x4xf16>
      %227 = amdgpu.mfma %226 * %206 + %225 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %228 = vector.insert %227, %218 [1, 0] : vector<16xf32> into vector<2x2x16xf32>
      %229 = vector.extract %arg1[1, 1] : vector<16xf32> from vector<2x2x16xf32>
      %230 = amdgpu.mfma %220 * %210 + %229 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %231 = amdgpu.mfma %222 * %212 + %230 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %232 = amdgpu.mfma %224 * %214 + %231 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %233 = amdgpu.mfma %226 * %216 + %232 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
      %234 = vector.insert %233, %228 [1, 1] : vector<16xf32> into vector<2x2x16xf32>
      gpu.barrier
      memref.dealloc %alloc_2 : memref<256x32xf16, #gpu.address_space<workgroup>>
      memref.dealloc %alloc : memref<128x32xf16, #gpu.address_space<workgroup>>
      scf.yield %234 : vector<2x2x16xf32>
    }
    %11 = vector.extract %10[0, 0, 0] : f32 from vector<2x2x16xf32>
    %12 = arith.remui %0, %c32 : index
    %13 = arith.divui %0, %c32 : index
    %14 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%13]
    %15 = arith.addi %14, %6 : index
    %16 = arith.addi %12, %7 : index
    memref.store %11, %5[%15, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %17 = vector.extract %10[0, 1, 0] :  f32 from vector<2x2x16xf32>
    %18 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%12]
    %19 = arith.addi %18, %7 : index
    memref.store %17, %5[%15, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %20 = vector.extract %10[0, 0, 1] :  f32 from vector<2x2x16xf32>
    %21 = affine.apply affine_map<()[s0] -> (s0 * 4 + 1)>()[%13]
    %22 = arith.addi %21, %6 : index
    memref.store %20, %5[%22, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %23 = vector.extract %10[0, 1, 1] :  f32 from vector<2x2x16xf32>
    memref.store %23, %5[%22, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %24 = vector.extract %10[0, 0, 2] :  f32 from vector<2x2x16xf32>
    %25 = affine.apply affine_map<()[s0] -> (s0 * 4 + 2)>()[%13]
    %26 = arith.addi %25, %6 : index
    memref.store %24, %5[%26, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %27 = vector.extract %10[0, 1, 2] :  f32 from  vector<2x2x16xf32>
    memref.store %27, %5[%26, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %28 = vector.extract %10[0, 0, 3] :  f32 from vector<2x2x16xf32>
    %29 = affine.apply affine_map<()[s0] -> (s0 * 4 + 3)>()[%13]
    %30 = arith.addi %29, %6 : index
    memref.store %28, %5[%30, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %31 = vector.extract %10[0, 1, 3] :  f32 from vector<2x2x16xf32>
    memref.store %31, %5[%30, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %32 = vector.extract %10[0, 0, 4] :  f32 from vector<2x2x16xf32>
    %33 = affine.apply affine_map<()[s0] -> (s0 * 4 + 8)>()[%13]
    %34 = arith.addi %33, %6 : index
    memref.store %32, %5[%34, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %35 = vector.extract %10[0, 1, 4] :  f32 from vector<2x2x16xf32>
    memref.store %35, %5[%34, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %36 = vector.extract %10[0, 0, 5] :  f32 from  vector<2x2x16xf32>
    %37 = affine.apply affine_map<()[s0] -> (s0 * 4 + 9)>()[%13]
    %38 = arith.addi %37, %6 : index
    memref.store %36, %5[%38, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %39 = vector.extract %10[0, 1, 5] :  f32 from  vector<2x2x16xf32>
    memref.store %39, %5[%38, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %40 = vector.extract %10[0, 0, 6] :  f32 from  vector<2x2x16xf32>
    %41 = affine.apply affine_map<()[s0] -> (s0 * 4 + 10)>()[%13]
    %42 = arith.addi %41, %6 : index
    memref.store %40, %5[%42, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %43 = vector.extract %10[0, 1, 6] :  f32 from  vector<2x2x16xf32>
    memref.store %43, %5[%42, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %44 = vector.extract %10[0, 0, 7] :  f32 from  vector<2x2x16xf32>
    %45 = affine.apply affine_map<()[s0] -> (s0 * 4 + 11)>()[%13]
    %46 = arith.addi %45, %6 : index
    memref.store %44, %5[%46, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %47 = vector.extract %10[0, 1, 7] :  f32 from  vector<2x2x16xf32>
    memref.store %47, %5[%46, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %48 = vector.extract %10[0, 0, 8] :  f32 from  vector<2x2x16xf32>
    %49 = affine.apply affine_map<()[s0] -> (s0 * 4 + 16)>()[%13]
    %50 = arith.addi %49, %6 : index
    memref.store %48, %5[%50, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %51 = vector.extract %10[0, 1, 8] :  f32 from  vector<2x2x16xf32>
    memref.store %51, %5[%50, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %52 = vector.extract %10[0, 0, 9] :  f32 from  vector<2x2x16xf32>
    %53 = affine.apply affine_map<()[s0] -> (s0 * 4 + 17)>()[%13]
    %54 = arith.addi %53, %6 : index
    memref.store %52, %5[%54, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %55 = vector.extract %10[0, 1, 9] :  f32 from  vector<2x2x16xf32>
    memref.store %55, %5[%54, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %56 = vector.extract %10[0, 0, 10] :  f32 from  vector<2x2x16xf32>
    %57 = affine.apply affine_map<()[s0] -> (s0 * 4 + 18)>()[%13]
    %58 = arith.addi %57, %6 : index
    memref.store %56, %5[%58, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %59 = vector.extract %10[0, 1, 10] :  f32 from  vector<2x2x16xf32>
    memref.store %59, %5[%58, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %60 = vector.extract %10[0, 0, 11] :  f32 from  vector<2x2x16xf32>
    %61 = affine.apply affine_map<()[s0] -> (s0 * 4 + 19)>()[%13]
    %62 = arith.addi %61, %6 : index
    memref.store %60, %5[%62, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %63 = vector.extract %10[0, 1, 11] :  f32 from  vector<2x2x16xf32>
    memref.store %63, %5[%62, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %64 = vector.extract %10[0, 0, 12] :  f32 from  vector<2x2x16xf32>
    %65 = affine.apply affine_map<()[s0] -> (s0 * 4 + 24)>()[%13]
    %66 = arith.addi %65, %6 : index
    memref.store %64, %5[%66, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %67 = vector.extract %10[0, 1, 12] :  f32 from  vector<2x2x16xf32>
    memref.store %67, %5[%66, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %68 = vector.extract %10[0, 0, 13] :  f32 from  vector<2x2x16xf32>
    %69 = affine.apply affine_map<()[s0] -> (s0 * 4 + 25)>()[%13]
    %70 = arith.addi %69, %6 : index
    memref.store %68, %5[%70, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %71 = vector.extract %10[0, 1, 13] :  f32 from  vector<2x2x16xf32>
    memref.store %71, %5[%70, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %72 = vector.extract %10[0, 0, 14] :  f32 from  vector<2x2x16xf32>
    %73 = affine.apply affine_map<()[s0] -> (s0 * 4 + 26)>()[%13]
    %74 = arith.addi %73, %6 : index
    memref.store %72, %5[%74, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %75 = vector.extract %10[0, 1, 14] :  f32 from  vector<2x2x16xf32>
    memref.store %75, %5[%74, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %76 = vector.extract %10[0, 0, 15] :  f32 from  vector<2x2x16xf32>
    %77 = affine.apply affine_map<()[s0] -> (s0 * 4 + 27)>()[%13]
    %78 = arith.addi %77, %6 : index
    memref.store %76, %5[%78, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %79 = vector.extract %10[0, 1, 15] :  f32 from  vector<2x2x16xf32>
    memref.store %79, %5[%78, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %80 = vector.extract %10[1, 0, 0] :  f32 from  vector<2x2x16xf32>
    %81 = affine.apply affine_map<()[s0] -> (s0 * 4 + 32)>()[%13]
    %82 = arith.addi %81, %6 : index
    memref.store %80, %5[%82, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %83 = vector.extract %10[1, 1, 0] :  f32 from  vector<2x2x16xf32>
    memref.store %83, %5[%82, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %84 = vector.extract %10[1, 0, 1] :  f32 from  vector<2x2x16xf32>
    %85 = affine.apply affine_map<()[s0] -> (s0 * 4 + 33)>()[%13]
    %86 = arith.addi %85, %6 : index
    memref.store %84, %5[%86, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %87 = vector.extract %10[1, 1, 1] :  f32 from  vector<2x2x16xf32>
    memref.store %87, %5[%86, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %88 = vector.extract %10[1, 0, 2] :  f32 from  vector<2x2x16xf32>
    %89 = affine.apply affine_map<()[s0] -> (s0 * 4 + 34)>()[%13]
    %90 = arith.addi %89, %6 : index
    memref.store %88, %5[%90, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %91 = vector.extract %10[1, 1, 2] :  f32 from  vector<2x2x16xf32>
    memref.store %91, %5[%90, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %92 = vector.extract %10[1, 0, 3] :  f32 from  vector<2x2x16xf32>
    %93 = affine.apply affine_map<()[s0] -> (s0 * 4 + 35)>()[%13]
    %94 = arith.addi %93, %6 : index
    memref.store %92, %5[%94, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %95 = vector.extract %10[1, 1, 3] :  f32 from  vector<2x2x16xf32>
    memref.store %95, %5[%94, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %96 = vector.extract %10[1, 0, 4] :  f32 from  vector<2x2x16xf32>
    %97 = affine.apply affine_map<()[s0] -> (s0 * 4 + 40)>()[%13]
    %98 = arith.addi %97, %6 : index
    memref.store %96, %5[%98, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %99 = vector.extract %10[1, 1, 4] :  f32 from  vector<2x2x16xf32>
    memref.store %99, %5[%98, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %100 = vector.extract %10[1, 0, 5] :  f32 from  vector<2x2x16xf32>
    %101 = affine.apply affine_map<()[s0] -> (s0 * 4 + 41)>()[%13]
    %102 = arith.addi %101, %6 : index
    memref.store %100, %5[%102, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %103 = vector.extract %10[1, 1, 5] :  f32 from  vector<2x2x16xf32>
    memref.store %103, %5[%102, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %104 = vector.extract %10[1, 0, 6] :  f32 from  vector<2x2x16xf32>
    %105 = affine.apply affine_map<()[s0] -> (s0 * 4 + 42)>()[%13]
    %106 = arith.addi %105, %6 : index
    memref.store %104, %5[%106, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %107 = vector.extract %10[1, 1, 6] :  f32 from  vector<2x2x16xf32>
    memref.store %107, %5[%106, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %108 = vector.extract %10[1, 0, 7] :  f32 from  vector<2x2x16xf32>
    %109 = affine.apply affine_map<()[s0] -> (s0 * 4 + 43)>()[%13]
    %110 = arith.addi %109, %6 : index
    memref.store %108, %5[%110, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %111 = vector.extract %10[1, 1, 7] :  f32 from  vector<2x2x16xf32>
    memref.store %111, %5[%110, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %112 = vector.extract %10[1, 0, 8] :  f32 from  vector<2x2x16xf32>
    %113 = affine.apply affine_map<()[s0] -> (s0 * 4 + 48)>()[%13]
    %114 = arith.addi %113, %6 : index
    memref.store %112, %5[%114, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %115 = vector.extract %10[1, 1, 8] :  f32 from  vector<2x2x16xf32>
    memref.store %115, %5[%114, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %116 = vector.extract %10[1, 0, 9] :  f32 from  vector<2x2x16xf32>
    %117 = affine.apply affine_map<()[s0] -> (s0 * 4 + 49)>()[%13]
    %118 = arith.addi %117, %6 : index
    memref.store %116, %5[%118, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %119 = vector.extract %10[1, 1, 9] :  f32 from  vector<2x2x16xf32>
    memref.store %119, %5[%118, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %120 = vector.extract %10[1, 0, 10] :  f32 from  vector<2x2x16xf32>
    %121 = affine.apply affine_map<()[s0] -> (s0 * 4 + 50)>()[%13]
    %122 = arith.addi %121, %6 : index
    memref.store %120, %5[%122, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %123 = vector.extract %10[1, 1, 10] :  f32 from  vector<2x2x16xf32>
    memref.store %123, %5[%122, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %124 = vector.extract %10[1, 0, 11] :  f32 from  vector<2x2x16xf32>
    %125 = affine.apply affine_map<()[s0] -> (s0 * 4 + 51)>()[%13]
    %126 = arith.addi %125, %6 : index
    memref.store %124, %5[%126, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %127 = vector.extract %10[1, 1, 11] :  f32 from  vector<2x2x16xf32>
    memref.store %127, %5[%126, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %128 = vector.extract %10[1, 0, 12] :  f32 from  vector<2x2x16xf32>
    %129 = affine.apply affine_map<()[s0] -> (s0 * 4 + 56)>()[%13]
    %130 = arith.addi %129, %6 : index
    memref.store %128, %5[%130, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %131 = vector.extract %10[1, 1, 12] :  f32 from  vector<2x2x16xf32>
    memref.store %131, %5[%130, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %132 = vector.extract %10[1, 0, 13] :  f32 from  vector<2x2x16xf32>
    %133 = affine.apply affine_map<()[s0] -> (s0 * 4 + 57)>()[%13]
    %134 = arith.addi %133, %6 : index
    memref.store %132, %5[%134, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %135 = vector.extract %10[1, 1, 13] :  f32 from  vector<2x2x16xf32>
    memref.store %135, %5[%134, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %136 = vector.extract %10[1, 0, 14] :  f32 from  vector<2x2x16xf32>
    %137 = affine.apply affine_map<()[s0] -> (s0 * 4 + 58)>()[%13]
    %138 = arith.addi %137, %6 : index
    memref.store %136, %5[%138, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %139 = vector.extract %10[1, 1, 14] :  f32 from  vector<2x2x16xf32>
    memref.store %139, %5[%138, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %140 = vector.extract %10[1, 0, 15] :  f32 from  vector<2x2x16xf32>
    %141 = affine.apply affine_map<()[s0] -> (s0 * 4 + 59)>()[%13]
    %142 = arith.addi %141, %6 : index
    memref.store %140, %5[%142, %16] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    %143 = vector.extract %10[1, 1, 15] : f32 from  vector<2x2x16xf32>
    memref.store %143, %5[%142, %19] : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
    return
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.optimize_shared_memory_reads_and_writes %func : (!transform.any_op) -> ()
    transform.yield
  } // @__transform_main
} // module

// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 256 + s2 * 64 + (s1 floordiv 64) * 64 - ((s2 + s1 floordiv 64) floordiv 4) * 256)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 128 + ((s2 + s1 floordiv 64) floordiv 4) * 64)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<()[s0, s1] -> (s1 * 64 + (s0 floordiv 64) * 64 - ((s1 + s0 floordiv 64) floordiv 4) * 256)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<()[s0, s1] -> (((s1 + s0 floordiv 64) floordiv 4) * 64)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 256 + s2 * 16 + s3 * 128 + s1 floordiv 4)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<()[s0, s1] -> (s0 + s1 * 8 - (s1 floordiv 4) * 32)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 128 + s0 floordiv 4)>
// CHECK-DAG:  #[[MAP7:.+]] = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>
// CHECK-DAG:  #[[MAP8:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 256 + s2 * 16 + s3 * 128 + s1 floordiv 4 + 128)>
// CHECK-DAG:  #[[MAP9:.+]] = affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 128 + s0 floordiv 4 + 128)>
// CHECK-DAG:  #[[MAP10:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 128 + s2 * 16 + s3 * 128 + s1 floordiv 4)>

// CHECK: func.func @matmul() {
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<2x2x16xf32>
// CHECK-DAG:    %[[C4096:.+]] = arith.constant 4096 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<2x2x8xf16>
// CHECK-DAG:    %[[D0:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D1:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D2:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x32xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:    %[[alloc_2:.+]] = memref.alloc() {alignment = 64 : i64} : memref<256x32xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:    %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[C0:.+]]) flags(ReadOnly) : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>
// CHECK         memref.assume_alignment %[[D3:.+]], 64 : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:    %[[D4:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[C0:.+]]) flags(ReadOnly) : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>
// CHECK         memref.assume_alignment %[[D4:.+]], 64 : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:    %[[D5:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%[[C0:.+]]) : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
// CHECK         memref.assume_alignment %[[D5:.+]], 64 : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:        %[[WORKGROUP_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-DAG:    %[[D6:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_X]], %[[D0]], %[[D1]]]
// CHECK-DAG:    %[[D7:.+]] = affine.apply #[[MAP1]]()[%[[WORKGROUP_ID_Y]], %[[D0]], %[[D1]]]
// CHECK-DAG:    %[[D8:.+]] = affine.apply #[[MAP2]]()[%[[D0]], %[[D1]]]
// CHECK-DAG:    %[[D9:.+]] = affine.apply #[[MAP3]]()[%[[D0]], %[[D1]]]
// CHECK:        %[[D10:.+]] = scf.for %[[ARG0:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C4096]] step %[[C32]] 
// CHECK-SAME:   iter_args(%[[ARG1:[a-zA-Z0-9_]+]] = %[[CST_0]]) 
// CHEKC-SAME:   -> (vector<2x2x16xf32>) {
// CHECK-DAG:    %[[D144:.+]] = affine.apply #[[MAP4]]()[%[[WORKGROUP_ID_X]], %[[D0]], %[[D1]], %[[D2]]]
// CHECK-DAG:    %[[D145:.+]] = affine.apply #[[MAP5]]()[%[[ARG0:[a-zA-Z0-9_]+]], %[[D0]]]
// CHECK:        %[[D146:.+]] = vector.transfer_read %[[D3]][%[[D144]], %[[D145]]], %[[CST]] {in_bounds = [true, true]} : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
// CHECK-DAG:    %[[D147:.+]] = affine.apply #[[MAP6]]()[%[[D0]], %[[D1]], %[[D2]]]
// CHECK-DAG:    %[[D148:.+]] = affine.apply #[[MAP7]]()[%[[D0]]]
// CHECK-DAG:    %[[C7:.+]] = arith.constant 7 : index
// CHECK-DAG:    %[[D149:.+]] = arith.andi %[[D147]], %[[C7]] : index
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[D150:.+]] = arith.shli %[[D149]], %[[C2]] : index
// CHECK-DAG:    %[[D151:.+]] = arith.xori %[[D148]], %[[D150]] : index
// CHECK:        vector.transfer_write %[[D146]], %[[alloc_2]][%[[D147]], %[[D151]]] {in_bounds = [true, true]} : vector<1x8xf16>, memref<256x32xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:    %[[D152:.+]] = affine.apply #[[MAP8]]()[%[[WORKGROUP_ID_X]], %[[D0]], %[[D1]], %[[D2]]]
// CHECK:        %[[D153:.+]] = vector.transfer_read %[[D3]][%[[D152]], %[[D145]]], %[[CST]] {in_bounds = [true, true]} : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
// CHECK-DAG:    %[[D154:.+]] = affine.apply #[[MAP9]]()[%[[D0]], %[[D1]], %[[D2]]]
// CHECK-DAG:    %[[C7_3:.+]] = arith.constant 7 : index
// CHECK-DAG:    %[[D155:.+]] = arith.andi %[[D154]], %[[C7_3]] : index
// CHECK-DAG:    %[[C2_4:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[D156:.+]]    = arith.shli %[[D155]], %[[C2_4]] : index
// CHECK-DAG:    %[[D157:.+]] = arith.xori %[[D148]], %[[D156]] : index
// CHECK:        vector.transfer_write %[[D153]], %[[alloc_2]][%[[D154]], %[[D157]]] {in_bounds = [true, true]} : vector<1x8xf16>, memref<256x32xf16, #gpu.address_space<workgroup>>
// CHECK-DAG:    %[[D158:.+]] = affine.apply #[[MAP10]]()[%[[WORKGROUP_ID_Y]], %[[D0]], %[[D1]], %[[D2]]]
// CHECK:        %[[D159:.+]] = vector.transfer_read %[[D4]][%[[D158]], %[[D145]]], %[[CST]] {in_bounds = [true, true]} : memref<4096x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
// CHECK-DAG:    %[[C7_5:.+]] = arith.constant 7 : index
// CHECK-DAG:    %[[D160:.+]] = arith.andi %[[D147]], %[[C7_5]] : index
// CHECK-DAG:    %[[C2_6:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[D161:.+]] = arith.shli %[[D160]], %[[C2_6]] : index
// CHECK-DAG:    %[[D162:.+]] = arith.xori %[[D148]], %[[D161]] : index
// CHECK:        vector.transfer_write %[[D159]], %[[alloc]][%[[D147]], %[[D162]]] {in_bounds = [true, true]} : vector<1x8xf16>, memref<128x32xf16, #gpu.address_space<workgroup>>
// CHECK:        gpu.barrier
// CHECK:        gpu.barrier
// CHECK: }
