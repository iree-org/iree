// RUN: iree-opt --split-input-file --iree-flow-deduplicate-executables %s | FileCheck %s

// CHECK-LABEL: flow.executable public @single_executable_ex_0
flow.executable @single_executable_ex_0 {
  flow.executable.export @single_executable_entry_0
  builtin.module {
    func.func @single_executable_entry_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func.func @single_executable
func.func @single_executable(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @single_executable_ex_0::@single_executable_entry_0[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @single_executable_ex_0::@single_executable_entry_0[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: flow.executable public @duplicate_executables_ex_0
flow.executable @duplicate_executables_ex_0 {
  flow.executable.export @duplicate_executables_entry_0
  builtin.module {
    func.func @duplicate_executables_entry_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-NOT: flow.executable public @duplicate_executables_ex_1
flow.executable @duplicate_executables_ex_1 {
  flow.executable.export @duplicate_executables_entry_1
  builtin.module {
    func.func @duplicate_executables_entry_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: flow.executable public @duplicate_executables_ex_2
flow.executable @duplicate_executables_ex_2 {
  flow.executable.export @duplicate_executables_entry_2
  builtin.module {
    func.func @duplicate_executables_entry_2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = arith.subf %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func.func @duplicate_executables
func.func @duplicate_executables(%arg0: tensor<4xf32>) {
  %c4 = arith.constant 4 : index
  // CHECK: = flow.dispatch @duplicate_executables_ex_0::@duplicate_executables_entry_0[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @duplicate_executables_ex_0::@duplicate_executables_entry_0[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: = flow.dispatch @duplicate_executables_ex_0::@duplicate_executables_entry_0[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = flow.dispatch @duplicate_executables_ex_1::@duplicate_executables_entry_1[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: = flow.dispatch @duplicate_executables_ex_2::@duplicate_executables_entry_2[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = flow.dispatch @duplicate_executables_ex_2::@duplicate_executables_entry_2[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: = flow.dispatch {@duplicate_executables_ex_0::@duplicate_executables_entry_0, @duplicate_executables_ex_0::@duplicate_executables_entry_0}
  %3 = flow.dispatch {@duplicate_executables_ex_0::@duplicate_executables_entry_0, @duplicate_executables_ex_1::@duplicate_executables_entry_1}[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return
}

// Ensure that symbol renaming is done within initializers.
// CHECK: util.initializer
util.initializer {
  // CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00>
  %cst = arith.constant dense<1.000000e+00> : tensor<4xf32>
  // CHECK: {{.*}} = flow.dispatch @duplicate_executables_ex_0::@duplicate_executables_entry_0(%[[CST]]) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @duplicate_executables_ex_1::@duplicate_executables_entry_1(%cst) : (tensor<4xf32>) -> tensor<4xf32>
  util.optimization_barrier %0 : tensor<4xf32>
  util.return
}

// -----

// CHECK: flow.executable public @same_ops_diff_operands_ex_0
flow.executable @same_ops_diff_operands_ex_0 {
  flow.executable.export @entry_0
  builtin.module {
    func.func @entry_0(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi32> {
      %0 = arith.muli %arg0, %arg1 : tensor<2xi32>
      return %0 : tensor<2xi32>
    }
  }
}
// CHECK: flow.executable public @same_ops_diff_operands_ex_1
flow.executable @same_ops_diff_operands_ex_1 {
  flow.executable.export @entry_1
  builtin.module {
    func.func @entry_1(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = arith.muli %arg0, %arg0 : tensor<2xi32>
      return %0 : tensor<2xi32>
    }
  }
}
// CHECK-LABEL: func.func @same_ops_diff_operands
func.func @same_ops_diff_operands(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @same_ops_diff_operands_ex_0::@entry_0[%c4](%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %0 = flow.dispatch @same_ops_diff_operands_ex_0::@entry_0[%c4](%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  // CHECK: %1 = flow.dispatch @same_ops_diff_operands_ex_1::@entry_1[%c4](%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %1 = flow.dispatch @same_ops_diff_operands_ex_1::@entry_1[%c4](%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: flow.executable public @multiple_entry_points_ex_0
flow.executable @multiple_entry_points_ex_0 {
  flow.executable.export @multiple_entry_points_0_entry_0
  flow.executable.export @multiple_entry_points_0_entry_1
  builtin.module {
    func.func @multiple_entry_points_0_entry_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
    func.func @multiple_entry_points_0_entry_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = arith.subf %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-NOT: flow.executable public @multiple_entry_points_ex_1
flow.executable @multiple_entry_points_ex_1 {
  flow.executable.export @multiple_entry_points_1_entry_0
  flow.executable.export @multiple_entry_points_1_entry_1
  builtin.module {
    func.func @multiple_entry_points_1_entry_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
    func.func @multiple_entry_points_1_entry_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = arith.subf %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func.func @multiple_entry_points
func.func @multiple_entry_points(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[C4:.*]] = arith.constant 4
  %c4 = arith.constant 4 : index
  // CHECK:      {{.*}} = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_0[%[[C4]]](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_0[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: {{.*}} = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_1[%[[C4]]](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_1[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: {{.*}} = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_0[%[[C4]]](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = flow.dispatch @multiple_entry_points_ex_1::@multiple_entry_points_1_entry_0[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: {{.*}} = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_1[%[[C4]]](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = flow.dispatch @multiple_entry_points_ex_1::@multiple_entry_points_1_entry_1[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: flow.executable public @different_types_float_ex
flow.executable @different_types_float_ex {
  flow.executable.export @different_types_float_entry
  builtin.module {
    func.func @different_types_float_entry(%arg0: tensor<4xf32>) -> tensor<4xi1> {
      %0 = arith.cmpf ueq, %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xi1>
    }
  }
}
// CHECK-LABEL: flow.executable public @different_types_int_ex
flow.executable @different_types_int_ex {
  flow.executable.export @different_types_int_entry
  builtin.module {
    func.func @different_types_int_entry(%arg0: tensor<4xi32>) -> tensor<4xi1> {
      %0 = arith.cmpi eq, %arg0, %arg0 : tensor<4xi32>
      return %0 : tensor<4xi1>
    }
  }
}
// CHECK-LABEL: func.func @different_types
func.func @different_types(%arg0: tensor<4xf32>) -> tensor<4xi1> {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @different_types_float_ex::@different_types_float_entry[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  %0 = flow.dispatch @different_types_float_ex::@different_types_float_entry[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  // CHECK: %1 = flow.dispatch @different_types_int_ex::@different_types_int_entry[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  %1 = flow.dispatch @different_types_int_ex::@different_types_int_entry[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: flow.executable public @nested_ops_ex_0
#map0 = affine_map<(d0, d1) -> (d0, d1)>
flow.executable @nested_ops_ex_0 {
  flow.executable.export @nested_ops_entry_0
  builtin.module {
    func.func @nested_ops_entry_0(%input0: tensor<5x6xf32>, %input1: tensor<5x6xf32>) -> tensor<5x6xf32> {
      %init = tensor.empty() : tensor<5x6xf32>
      %max = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%input0, %input1 : tensor<5x6xf32>, tensor<5x6xf32>) outs(%init : tensor<5x6xf32>) {
      ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
        %27 = arith.maximumf %arg1, %arg2 : f32
        linalg.yield %27 : f32
      } -> tensor<5x6xf32>
      return %max : tensor<5x6xf32>
    }
  }
}
// CHECK-NOT: flow.executable public @nested_ops_ex_1
flow.executable @nested_ops_ex_1 {
  flow.executable.export @nested_ops_entry_1
  builtin.module {
    func.func @nested_ops_entry_1(%input0: tensor<5x6xf32>, %input1: tensor<5x6xf32>) -> tensor<5x6xf32> {
      %init = tensor.empty() : tensor<5x6xf32>
      %max = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%input0, %input1 : tensor<5x6xf32>, tensor<5x6xf32>) outs(%init : tensor<5x6xf32>) {
      ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
        %27 = arith.maximumf %arg1, %arg2 : f32
        linalg.yield %27 : f32
      } -> tensor<5x6xf32>
      return %max : tensor<5x6xf32>
    }
  }
}
// CHECK-LABEL: flow.executable public @nested_ops_ex_2
flow.executable @nested_ops_ex_2 {
  flow.executable.export @nested_ops_entry_2
  builtin.module {
    func.func @nested_ops_entry_2(%input0: tensor<5x6xf32>, %input1: tensor<5x6xf32>) -> tensor<5x6xf32> {
      %init = tensor.empty() : tensor<5x6xf32>
      %min = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%input0, %input1 : tensor<5x6xf32>, tensor<5x6xf32>) outs(%init : tensor<5x6xf32>) {
      ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
        %27 = arith.minimumf %arg1, %arg2 : f32
        linalg.yield %27 : f32
      } -> tensor<5x6xf32>
      return %min : tensor<5x6xf32>
    }
  }
}
// CHECK-LABEL: func.func @nested_ops
func.func @nested_ops(%arg0: tensor<5x6xf32>, %arg1: tensor<5x6xf32>) -> tensor<5x6xf32> {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @nested_ops_ex_0::@nested_ops_entry_0[%c4](%arg0, %arg1) : (tensor<5x6xf32>, tensor<5x6xf32>) -> tensor<5x6xf32>
  %0 = flow.dispatch @nested_ops_ex_0::@nested_ops_entry_0[%c4](%arg0, %arg1) : (tensor<5x6xf32>, tensor<5x6xf32>) -> tensor<5x6xf32>
  // CHECK: %1 = flow.dispatch @nested_ops_ex_0::@nested_ops_entry_0[%c4](%arg0, %arg1) : (tensor<5x6xf32>, tensor<5x6xf32>) -> tensor<5x6xf32>
  %1 = flow.dispatch @nested_ops_ex_0::@nested_ops_entry_0[%c4](%arg0, %arg1) : (tensor<5x6xf32>, tensor<5x6xf32>) -> tensor<5x6xf32>
  // CHECK: %2 = flow.dispatch @nested_ops_ex_2::@nested_ops_entry_2[%c4](%arg0, %arg1) : (tensor<5x6xf32>, tensor<5x6xf32>) -> tensor<5x6xf32>
  %2 = flow.dispatch @nested_ops_ex_2::@nested_ops_entry_2[%c4](%arg0, %arg1) : (tensor<5x6xf32>, tensor<5x6xf32>) -> tensor<5x6xf32>
  return %0 : tensor<5x6xf32>
}

// -----


// CHECK-LABEL: flow.executable public @attributes_ex_0
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
flow.executable @attributes_ex_0 {
  flow.executable.export @attributes_entry_0
  builtin.module {
    func.func @attributes_entry_0(%input0: tensor<5x6xf32>, %input1: tensor<5x6xf32>) -> tensor<5x6xf32> {
      %init = tensor.empty() : tensor<5x6xf32>
      %max = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%input0, %input1 : tensor<5x6xf32>, tensor<5x6xf32>) outs(%init : tensor<5x6xf32>) {
      ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
        %27 = arith.maximumf %arg1, %arg2 : f32
        linalg.yield %27 : f32
      } -> tensor<5x6xf32>
      return %max : tensor<5x6xf32>
    }
  }
}
// CHECK-LABEL: flow.executable public @attributes_ex_1
flow.executable @attributes_ex_1 {
  flow.executable.export @attributes_entry_1
  builtin.module {
    func.func @attributes_entry_1(%input0: tensor<5x6xf32>, %input1: tensor<5x6xf32>) -> tensor<5x6xf32> {
      %init = tensor.empty() : tensor<5x6xf32>
      // map1 instead of map0
      %max = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%input0, %input1 : tensor<5x6xf32>, tensor<5x6xf32>) outs(%init : tensor<5x6xf32>) {
      ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
        %27 = arith.maximumf %arg1, %arg2 : f32
        linalg.yield %27 : f32
      } -> tensor<5x6xf32>
      return %max : tensor<5x6xf32>
    }
  }
}
// Duplicate of @attributes_ex_0
// CHECK-NOT: flow.executable public @attributes_ex_2
flow.executable @attributes_ex_2 {
  flow.executable.export @attributes_entry_2
  builtin.module {
    func.func @attributes_entry_2(%input0: tensor<5x6xf32>, %input1: tensor<5x6xf32>) -> tensor<5x6xf32> {
      %init = tensor.empty() : tensor<5x6xf32>
      %max = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%input0, %input1 : tensor<5x6xf32>, tensor<5x6xf32>) outs(%init : tensor<5x6xf32>) {
      ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
        %27 = arith.maximumf %arg1, %arg2 : f32
        linalg.yield %27 : f32
      } -> tensor<5x6xf32>
      return %max : tensor<5x6xf32>
    }
  }
}

// -----

// Executable contents are the same but the workgroup count function of ex_1
// differs and should prevent it from deduplicating.
// Ideally we'd still deduplicate but add another export.

// CHECK-LABEL: flow.executable public @workgroup_count_ex_0
flow.executable @workgroup_count_ex_0 {
  flow.executable.export @workgroup_count_entry_0 workgroups(%arg0: index) -> (index, index, index) {
    flow.return %arg0, %arg0, %arg0 : index, index, index
  }
  builtin.module {
    func.func @workgroup_count_entry_0(%input: tensor<1xi32>) -> tensor<1xi32> {
      return %input : tensor<1xi32>
    }
  }
}

// CHECK-LABEL: flow.executable public @workgroup_count_ex_1
flow.executable @workgroup_count_ex_1 {
  flow.executable.export @workgroup_count_entry_1 workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    flow.return %arg0, %arg1, %arg2 : index, index, index
  }
  builtin.module {
    func.func @workgroup_count_entry_1(%input: tensor<1xi32>) -> tensor<1xi32> {
      return %input : tensor<1xi32>
    }
  }
}
// Duplicate of @workgroup_count_ex_0
// CHECK-NOT: flow.executable public @workgroup_count_ex_2
flow.executable @workgroup_count_ex_2 {
  flow.executable.export @workgroup_count_entry_2 workgroups(%arg0: index) -> (index, index, index) {
    flow.return %arg0, %arg0, %arg0 : index, index, index
  }
  builtin.module {
    func.func @workgroup_count_entry_2(%input: tensor<1xi32>) -> tensor<1xi32> {
      return %input : tensor<1xi32>
    }
  }
}

// -----

// CHECK-LABEL: flow.executable public @block_successors_ex_0
flow.executable @block_successors_ex_0 {
  flow.executable.export @entry_0
  builtin.module {
    func.func @entry_0(%arg0: i32, %arg1: i32) -> i32 {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %eqz = arith.cmpi eq, %arg0, %arg1 : i32
      cf.cond_br %eqz, ^bb_a(%c0 : i32), ^bb_b(%c1 : i32)
    ^bb_a(%bb_a_arg0 : i32):
      return %bb_a_arg0 : i32
    ^bb_b(%bb_b_arg0 : i32):
      return %bb_b_arg0 : i32
    }
  }
}
// CHECK-LABEL: flow.executable public @block_successors_ex_with_swapped_cond_br
flow.executable @block_successors_ex_with_swapped_cond_br {
  flow.executable.export @entry_1
  builtin.module {
    func.func @entry_0(%arg0: i32, %arg1: i32) -> i32 {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %eqz = arith.cmpi eq, %arg0, %arg1 : i32
      cf.cond_br %eqz, ^bb_b(%c0 : i32), ^bb_b(%c1 : i32)
    ^bb_a(%bb_a_arg0 : i32):
      return %bb_a_arg0 : i32
    ^bb_b(%bb_b_arg0 : i32):
      return %bb_b_arg0 : i32
    }
  }
}

// -----

// CHECK: hal.executable private @ex0
hal.executable private @ex0 {
  hal.executable.variant public @variant target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">) {
    hal.executable.condition(%device: !hal.device) -> i1 {
      %ok, %selected = hal.device.query<%device : !hal.device> key("some" :: "feature") : i1, i1
      hal.return %selected : i1
    }
    hal.executable.export public @dispatch ordinal(0)
        layout(#hal.pipeline.layout<push_constants = 0, sets = [
          <0, bindings = [
              <0, storage_buffer, ReadOnly>,
              <1, storage_buffer>
          ]>
        ]>) {
    ^bb0(%device: !hal.device, %workload: index):
      hal.return %workload, %workload, %workload : index, index, index
    }
  }
}
// CHECK-NOT: hal.executable private @ex1
hal.executable private @ex1 {
  hal.executable.variant public @variant target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">) {
    hal.executable.condition(%device: !hal.device) -> i1 {
      %ok, %selected = hal.device.query<%device : !hal.device> key("some" :: "feature") : i1, i1
      hal.return %selected : i1
    }
    hal.executable.export public @dispatch ordinal(0)
        layout(#hal.pipeline.layout<push_constants = 0, sets = [
          <0, bindings = [
              <0, storage_buffer, ReadOnly>,
              <1, storage_buffer>
          ]>
        ]>) {
    ^bb0(%device: !hal.device, %workload: index):
      hal.return %workload, %workload, %workload : index, index, index
    }
  }
}

// CHECK-LABEL: func.func @dispatch_variants
func.func @dispatch_variants(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[C4:.*]] = arith.constant 4
  %c4 = arith.constant 4 : index
  // CHECK:      {{.*}} = flow.dispatch @ex0::@variant::@dispatch[%[[C4]]](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @ex0::@variant::@dispatch[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: {{.*}} = flow.dispatch @ex0::@variant::@dispatch[%[[C4]]](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = flow.dispatch @ex1::@variant::@dispatch[%c4](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}
