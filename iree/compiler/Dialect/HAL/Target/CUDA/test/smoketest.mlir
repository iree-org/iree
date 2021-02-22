// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline -iree-hal-target-backends=cuda %s | IreeFileCheck %s


#map = affine_map<(d0) -> (d0)>
module  {
  flow.executable @add_dispatch_0 attributes {sym_visibility = "private"} {
    flow.dispatch.entry @add_dispatch_0 attributes {signature = (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>, workgroup_rank = 3 : index}
    module  {
      func @add_dispatch_0(%arg0: !flow.dispatch.input<16xf32>, %arg1: !flow.dispatch.input<16xf32>, %arg2: !flow.dispatch.output<16xf32>) {
        %0 = linalg.init_tensor [16] : tensor<16xf32>
        %1 = flow.dispatch.input.load %arg0 : !flow.dispatch.input<16xf32> -> tensor<16xf32>
        %2 = flow.dispatch.input.load %arg1 : !flow.dispatch.input<16xf32> -> tensor<16xf32>
        %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%1, %2 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) {
        ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
          %4 = addf %arg3, %arg4 : f32
          linalg.yield %4 : f32
        } -> tensor<16xf32>
        flow.dispatch.output.store %3, %arg2 : tensor<16xf32> -> !flow.dispatch.output<16xf32>
        return
      }
    }
  }
  func @add(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> attributes {iree.module.export, iree.reflection = {f = "I13!B4!d16B4!d16R7!B4!d16", fv = "1"}} {
    %c1 = constant 1 : index
    %c16 = constant 16 : index
    %0 = flow.ex.stream.fragment(%arg2 = %c16 : index, %arg3 = %c1 : index, %arg4 = %arg0 : tensor<16xf32>, %arg5 = %arg1 : tensor<16xf32>) -> tensor<16xf32> {
      %1 = flow.dispatch @add_dispatch_0::@add_dispatch_0[%arg2, %arg3, %arg3] (%arg4, %arg5) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
      flow.return %1 : tensor<16xf32>
    }
    return %0 : tensor<16xf32>
  }
}

//      CHECK:   hal.executable.binary @cuda attributes {
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = 1129661505 : i32}
