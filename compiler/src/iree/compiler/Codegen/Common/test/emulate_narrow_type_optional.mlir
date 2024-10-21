// RUN: iree-opt --split-input-file --iree-codegen-emulate-narrow-type --iree-enable-i1 %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @i1_datatype_emulation() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<8xi1, strided<[1], offset: ?>>
  %3 = vector.load %0[%c0] : memref<8xi1, strided<[1], offset: ?>>, vector<6xi1>
  %4 = vector.load %0[%c0] : memref<8xi1, strided<[1], offset: ?>>, vector<6xi1>
  %5 = arith.addi %3, %4 : vector<6xi1>
  vector.store %5, %0[%c0] : memref<8xi1, strided<[1], offset: ?>>, vector<6xi1>
  return
}
// CHECK-LABEL: @i1_datatype_emulation


// CHECK:      %[[EMU_LOAD:.+]] = vector.load
// CHECK-SAME: vector<1xi8>
// CHECK:      %[[BITCAST:.+]] = vector.bitcast %[[EMU_LOAD]]
// CHECK-SAME: vector<1xi8> to vector<8xi1>
// CHECK:      vector.extract_strided_slice %[[BITCAST]]
// CHECK-SAME: vector<8xi1> to vector<6xi1>

// CHECK:      %[[INSERT:.+]] = vector.insert_strided_slice
// CHECK-SAME: vector<6xi1> into vector<8xi1>
// CHECK:      vector.create_mask
// CHECK-SAME: vector<8xi1>

// CHECK:      vector.maskedstore
// CHECK-SAME: vector<1xi1>, vector<1xi8>

