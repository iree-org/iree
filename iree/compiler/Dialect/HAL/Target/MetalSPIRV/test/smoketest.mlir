// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline -iree-hal-target-backends=metal-spirv %s | IreeFileCheck %s

flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
      workload = 4 : index
  }
  module {
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: hal.executable @simpleMath_ex_dispatch_0
// CHECK-NEXT:   hal.interface @legacy_io {
// CHECK-DAG:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// CHECK-DAG:      hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
// CHECK-NEXT:   }
// CHECK-NEXT:   hal.executable.binary @metal_spirv attributes {
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = "MTLE"
