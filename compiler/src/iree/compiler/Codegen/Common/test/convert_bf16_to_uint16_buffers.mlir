// RUN: iree-opt --split-input-file \
// RUN:   --iree-convert-bf16-to-uint16-buffers %s | FileCheck %s

// CHECK-LABEL: @bf16_conversion
func.func @bf16_conversion() {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index

  // CHECK-DAG: %[[BUF0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<?xi16, #spirv.storage_class<StorageBuffer>>{%c8}
  // CHECK-DAG: %[[BUF1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<?xi16, #spirv.storage_class<StorageBuffer>>{%c8}
  // CHECK-DAG: %[[BUF2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<?xi16, #spirv.storage_class<StorageBuffer>>{%c8}
  // CHECK-DAG: %[[LOAD0:.+]] = memref.load %[[BUF0]][%arg0] : memref<?xi16, #spirv.storage_class<StorageBuffer>>
  // CHECK-DAG: %[[LOAD1:.+]] = memref.load %[[BUF1]][%arg0] : memref<?xi16, #spirv.storage_class<StorageBuffer>>
  // CHECK: memref.store %{{.+}}, %[[BUF2]][%arg0] : memref<?xi16, #spirv.storage_class<StorageBuffer>>
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<?xbf16, #spirv.storage_class<StorageBuffer>>{%c8}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<?xbf16, #spirv.storage_class<StorageBuffer>>{%c8}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<?xbf16, #spirv.storage_class<StorageBuffer>>{%c8}
  %3 = gpu.thread_id  x
  %4 = gpu.block_dim  x
  scf.for %arg0 = %3 to %c8 step %4 {
    %5 = memref.load %0[%arg0] : memref<?xbf16, #spirv.storage_class<StorageBuffer>>
    %6 = memref.load %1[%arg0] : memref<?xbf16, #spirv.storage_class<StorageBuffer>>
    %7 = arith.extf %5 : bf16 to f32
    %8 = arith.extf %6 : bf16 to f32
    %9 = arith.addf %7, %8 : f32
    %10 = arith.truncf %9 : f32 to bf16
    memref.store %10, %2[%arg0] : memref<?xbf16, #spirv.storage_class<StorageBuffer>>
  }
  return
}

// -----

// CHECK-LABEL: @bf16_constant
func.func @bf16_constant(%arg0 : bf16) -> bf16 {
  // CHECK: %[[CNST:.+]] = arith.constant 16256 : i16
  %c0 = arith.constant 1.0 : bf16
  // CHECK: return %[[CNST]]
  return %c0 : bf16
}

// -----

// CHECK-LABEL: @iree_uk_mmt4d
// CHECK-SAME:    memref<i16>
// CHECK-SAME:    memref<i16>
// CHECK-SAME:    memref<f32>
func.func private @iree_uk_mmt4d(memref<bf16>, index, index, memref<bf16>, index, index, memref<f32>, index, index, index, index, index, i32, i32, i32, i32) attributes {hal.import.bitcode = true, hal.import.fields = ["processor_data"], llvm.bareptr = true}

// CHECK-LABEL: @mmt4d_bf16xbf16xf32
// CHECK:         func.call
// CHECK-SAME:    memref<i16>
// CHECK-SAME:    memref<i16>
// CHECK-SAME:    memref<f32>
func.func @mmt4d_bf16xbf16xf32() {
  %c32 = arith.constant 32 : index
  %c24 = arith.constant 24 : index
  %c3 = arith.constant 3 : index
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1029_i32 = arith.constant 1029 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<1x3x8x1xbf16>
  memref.assume_alignment %0, 64 : memref<1x3x8x1xbf16>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c64) flags(ReadOnly) : memref<1x3x8x1xbf16, strided<[24, 8, 1, 1], offset: 32>>
  memref.assume_alignment %1, 64 : memref<1x3x8x1xbf16, strided<[24, 8, 1, 1], offset: 32>>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c128) : memref<1x1x8x8xf32, strided<[64, 64, 8, 1], offset: 32>>
  memref.assume_alignment %2, 64 : memref<1x1x8x8xf32, strided<[64, 64, 8, 1], offset: 32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  scf.for %arg0 = %workgroup_id_y to %c1 step %workgroup_count_y {
    scf.for %arg1 = %workgroup_id_x to %c1 step %workgroup_count_x {
      %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %0 : memref<1x3x8x1xbf16> -> memref<bf16>, index, index, index, index, index, index, index, index, index
      %base_buffer_0, %offset_1, %sizes_2:4, %strides_3:4 = memref.extract_strided_metadata %1 : memref<1x3x8x1xbf16, strided<[24, 8, 1, 1], offset: 32>> -> memref<bf16>, index, index, index, index, index, index, index, index, index
      %base_buffer_4, %offset_5, %sizes_6:4, %strides_7:4 = memref.extract_strided_metadata %2 : memref<1x1x8x8xf32, strided<[64, 64, 8, 1], offset: 32>> -> memref<f32>, index, index, index, index, index, index, index, index, index
      func.call @iree_uk_mmt4d(%base_buffer, %c0, %c24, %base_buffer_0, %c32, %c24, %base_buffer_4, %c32, %c64, %c1, %c1, %c3, %c8_i32, %c8_i32, %c1_i32, %c1029_i32) : (memref<bf16>, index, index, memref<bf16>, index, index, memref<f32>, index, index, index, index, index, i32, i32, i32, i32) -> ()
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @outerproduct_bf16_preserved
func.func @outerproduct_bf16_preserved(%arg0 : vector<1xbf16>, %arg1 : vector<1xbf16>, %arg2 : vector<1x1xbf16>) -> vector<1x1xbf16> {
  // CHECK: vector.outerproduct %[[ARG0:.+]], %[[ARG1:.+]], %[[ARG2:.+]] {kind = #vector.kind<add>} : vector<1xbf16>, vector<1xbf16>
  %0 = vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<add>} : vector<1xbf16>, vector<1xbf16>
  return %0 : vector<1x1xbf16>
}

// -----

// CHECK-LABEL: func.func @load_trunc_f32_bf16
func.func @load_trunc_f32_bf16(%arg0 : memref<32xf32>, %arg1 : memref<32xbf16>) {
  // CHECK-SAME:  %[[ARG0:.+]]: memref<32xf32>
  // CHECK-SAME:  %[[ARG1:.+]]: memref<32xi16>
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[LOAD:.+]] = vector.load %[[ARG0]][%[[C0]]] : memref<32xf32>, vector<4xf32>
  // CHECK: %[[TRUNC:.+]] = arith.truncf %[[LOAD]] : vector<4xf32> to vector<4xbf16>
  // CHECK: %[[CAST:.+]] = arith.bitcast %[[TRUNC]] : vector<4xbf16> to vector<4xi16>
  // CHECK: vector.store %[[CAST]], %[[ARG1]][%[[C0]]] : memref<32xi16>, vector<4xi16>
  %c0 = arith.constant 0 : index
  %load = vector.load %arg0[%c0] : memref<32xf32>, vector<4xf32>
  %trunc = arith.truncf %load : vector<4xf32> to vector<4xbf16>
  vector.store %trunc, %arg1[%c0] : memref<32xbf16>, vector<4xbf16>
  return
}
