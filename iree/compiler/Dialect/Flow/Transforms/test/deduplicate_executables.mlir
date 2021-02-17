// RUN: iree-opt -split-input-file -iree-flow-deduplicate-executables %s | IreeFileCheck %s

// CHECK-LABEL: flow.executable @single_executable_ex_0
flow.executable @single_executable_ex_0 {
  flow.dispatch.entry @single_executable_entry_0
  module {
    func @single_executable_entry_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func @single_executable
func @single_executable(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c4 = constant 4 : index
  // CHECK: %0 = flow.dispatch @single_executable_ex_0::@single_executable_entry_0[%c4](%arg0) : (tensor<4xf32>) -> (tensor<4xf32>)
  %0 = flow.dispatch @single_executable_ex_0::@single_executable_entry_0[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: flow.executable @duplicate_executables_ex_0
flow.executable @duplicate_executables_ex_0 {
  flow.dispatch.entry @duplicate_executables_entry_0
  module {
    func @duplicate_executables_entry_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-NOT: flow.executable @duplicate_executables_ex_1
flow.executable @duplicate_executables_ex_1 {
  flow.dispatch.entry @duplicate_executables_entry_1
  module {
    func @duplicate_executables_entry_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: flow.executable @duplicate_executables_ex_2
flow.executable @duplicate_executables_ex_2 {
  flow.dispatch.entry @duplicate_executables_entry_2
  module {
    func @duplicate_executables_entry_2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.subtract %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func @duplicate_executables
func @duplicate_executables(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c4 = constant 4 : index
  // CHECK: %0 = flow.dispatch @duplicate_executables_ex_0::@duplicate_executables_entry_0[%c4](%arg0) : (tensor<4xf32>) -> (tensor<4xf32>)
  %0 = flow.dispatch @duplicate_executables_ex_0::@duplicate_executables_entry_0[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %1 = flow.dispatch @duplicate_executables_ex_0::@duplicate_executables_entry_0[%c4](%arg0) : (tensor<4xf32>) -> (tensor<4xf32>)
  %1 = flow.dispatch @duplicate_executables_ex_1::@duplicate_executables_entry_1[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %2 = flow.dispatch @duplicate_executables_ex_2::@duplicate_executables_entry_2[%c4](%arg0) : (tensor<4xf32>) -> (tensor<4xf32>)
  %2 = flow.dispatch @duplicate_executables_ex_2::@duplicate_executables_entry_2[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK: flow.executable @same_ops_diff_operands_ex_0
flow.executable @same_ops_diff_operands_ex_0 {
  flow.dispatch.entry @entry_0
  module {
    func @entry_0(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi32> {
      %0 = mhlo.multiply %arg0, %arg1 : tensor<2xi32>
      return %0 : tensor<2xi32>
    }
  }
}
// CHECK: flow.executable @same_ops_diff_operands_ex_1
flow.executable @same_ops_diff_operands_ex_1 {
  flow.dispatch.entry @entry_1
  module {
    func @entry_1(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = mhlo.multiply %arg0, %arg0 : tensor<2xi32>
      return %0 : tensor<2xi32>
    }
  }
}
// CHECK-LABEL: func @same_ops_diff_operands
func @same_ops_diff_operands(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  %c4 = constant 4 : index
  // CHECK: %0 = flow.dispatch @same_ops_diff_operands_ex_0::@entry_0[%c4](%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> (tensor<2xi32>)
  %0 = flow.dispatch @same_ops_diff_operands_ex_0::@entry_0[%c4] (%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  // CHECK: %1 = flow.dispatch @same_ops_diff_operands_ex_1::@entry_1[%c4](%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> (tensor<2xi32>)
  %1 = flow.dispatch @same_ops_diff_operands_ex_1::@entry_1[%c4] (%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: flow.executable @multiple_entry_points_ex_0
flow.executable @multiple_entry_points_ex_0 {
  flow.dispatch.entry @multiple_entry_points_0_entry_0
  flow.dispatch.entry @multiple_entry_points_0_entry_1
  module {
    func @multiple_entry_points_0_entry_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
    func @multiple_entry_points_0_entry_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.subtract %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-NOT: flow.executable @multiple_entry_points_ex_1
flow.executable @multiple_entry_points_ex_1 {
  flow.dispatch.entry @multiple_entry_points_1_entry_0
  flow.dispatch.entry @multiple_entry_points_1_entry_1
  module {
    func @multiple_entry_points_1_entry_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
    func @multiple_entry_points_1_entry_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.subtract %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func @multiple_entry_points
func @multiple_entry_points(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c4 = constant 4 : index
  // CHECK: %0 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_0[%c4](%arg0) : (tensor<4xf32>) -> (tensor<4xf32>)
  %0 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_0[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %1 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_1[%c4](%arg0) : (tensor<4xf32>) -> (tensor<4xf32>)
  %1 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_1[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %2 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_0[%c4](%arg0) : (tensor<4xf32>) -> (tensor<4xf32>)
  %2 = flow.dispatch @multiple_entry_points_ex_1::@multiple_entry_points_1_entry_0[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %3 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_1[%c4](%arg0) : (tensor<4xf32>) -> (tensor<4xf32>)
  %3 = flow.dispatch @multiple_entry_points_ex_1::@multiple_entry_points_1_entry_1[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: flow.executable @different_types_float_ex
flow.executable @different_types_float_ex {
  flow.dispatch.entry @different_types_float_entry
  module {
    func @different_types_float_entry(%arg0: tensor<4xf32>) -> tensor<4xi1> {
      %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = "EQ"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      return %0 : tensor<4xi1>
    }
  }
}
// CHECK-LABEL: flow.executable @different_types_int_ex
flow.executable @different_types_int_ex {
  flow.dispatch.entry @different_types_int_entry
  module {
    func @different_types_int_entry(%arg0: tensor<4xi32>) -> tensor<4xi1> {
      %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = "EQ"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
      return %0 : tensor<4xi1>
    }
  }
}
// CHECK-LABEL: func @different_types
func @different_types(%arg0: tensor<4xf32>) -> tensor<4xi1> {
  %c4 = constant 4 : index
  // CHECK: %0 = flow.dispatch @different_types_float_ex::@different_types_float_entry[%c4](%arg0) : (tensor<4xf32>) -> (tensor<4xi1>)
  %0 = flow.dispatch @different_types_float_ex::@different_types_float_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  // CHECK: %1 = flow.dispatch @different_types_int_ex::@different_types_int_entry[%c4](%arg0) : (tensor<4xf32>) -> (tensor<4xi1>)
  %1 = flow.dispatch @different_types_int_ex::@different_types_int_entry[%c4] (%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: flow.executable @nested_ops_ex_0
flow.executable @nested_ops_ex_0 {
  flow.dispatch.entry @nested_ops_entry_0
  module {
    func @nested_ops_entry_0(%input: tensor<1x4xi32>) -> tensor<1xi32> {
      %0 = constant dense<0> : tensor<i32>
      %1 = "mhlo.reduce"(%input, %0) ( {
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
        %3 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%3) : (tensor<i32>) -> ()
      }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
      return %1 : tensor<1xi32>
    }
  }
}
// CHECK-NOT: flow.executable @nested_ops_ex_1
flow.executable @nested_ops_ex_1 {
  flow.dispatch.entry @nested_ops_entry_1
  module {
    func @nested_ops_entry_1(%input: tensor<1x4xi32>) -> tensor<1xi32> {
      %0 = constant dense<0> : tensor<i32>
      %1 = "mhlo.reduce"(%input, %0) ( {
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
        %3 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%3) : (tensor<i32>) -> ()
      }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
      return %1 : tensor<1xi32>
    }
  }
}
// CHECK-LABEL: flow.executable @nested_ops_ex_2
flow.executable @nested_ops_ex_2 {
  flow.dispatch.entry @nested_ops_entry_2
  module {
    func @nested_ops_entry_2(%input: tensor<1x4xi32>) -> tensor<1xi32> {
      %0 = constant dense<0> : tensor<i32>
      %1 = "mhlo.reduce"(%input, %0) ( {
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
        %3 = "mhlo.maximum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%3) : (tensor<i32>) -> ()
      }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
      return %1 : tensor<1xi32>
    }
  }
}
// CHECK-LABEL: func @nested_ops
func @nested_ops(%arg0: tensor<1x4xi32>) -> tensor<1xi32> {
  %c4 = constant 4 : index
  // CHECK: %0 = flow.dispatch @nested_ops_ex_0::@nested_ops_entry_0[%c4](%arg0) : (tensor<1x4xi32>) -> (tensor<1xi32>)
  %0 = flow.dispatch @nested_ops_ex_0::@nested_ops_entry_0[%c4] (%arg0) : (tensor<1x4xi32>) -> tensor<1xi32>
  // CHECK: %1 = flow.dispatch @nested_ops_ex_0::@nested_ops_entry_0[%c4](%arg0) : (tensor<1x4xi32>) -> (tensor<1xi32>)
  %1 = flow.dispatch @nested_ops_ex_0::@nested_ops_entry_0[%c4] (%arg0) : (tensor<1x4xi32>) -> tensor<1xi32>
  // CHECK: %2 = flow.dispatch @nested_ops_ex_2::@nested_ops_entry_2[%c4](%arg0) : (tensor<1x4xi32>) -> (tensor<1xi32>)
  %2 = flow.dispatch @nested_ops_ex_2::@nested_ops_entry_2[%c4] (%arg0) : (tensor<1x4xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// -----

// CHECK-LABEL: flow.executable @attributes_ex_0
flow.executable @attributes_ex_0 {
  flow.dispatch.entry @attributes_entry_0
  module {
    func @attributes_entry_0(%input: tensor<1x4xi32>) -> tensor<1xi32> {
      %0 = constant dense<0> : tensor<i32>
      %1 = "mhlo.reduce"(%input, %0) ( {
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
        %3 = "mhlo.maximum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%3) : (tensor<i32>) -> ()
      }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
      return %1 : tensor<1xi32>
    }
  }
}

// CHECK-LABEL: flow.executable @attributes_ex_1
flow.executable @attributes_ex_1 {
  flow.dispatch.entry @attributes_entry_1
  module {
    func @attributes_entry_1(%input: tensor<1x4xi32>) -> tensor<1xi32> {
      %0 = constant dense<0> : tensor<i32>
      %1 = "mhlo.reduce"(%input, %0) ( {
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
        %3 = "mhlo.maximum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%3) : (tensor<i32>) -> ()
        // @attributes_ex_0 but with a different attribute.
      }) {dimensions = dense<2> : tensor<1xi64>} : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
      return %1 : tensor<1xi32>
    }
  }
}
// Duplicate of @attributes_ex_0
// CHECK-NOT: flow.executable @attributes_ex_2
flow.executable @attributes_ex_2 {
  flow.dispatch.entry @attributes_entry_2
  module {
    func @attributes_entry_2(%input: tensor<1x4xi32>) -> tensor<1xi32> {
      %0 = constant dense<0> : tensor<i32>
      %1 = "mhlo.reduce"(%input, %0) ( {
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
        %3 = "mhlo.maximum"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%3) : (tensor<i32>) -> ()
      }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
      return %1 : tensor<1xi32>
    }
  }
}

// -----

// CHECK-LABEL: flow.executable @block_successors_ex_0
flow.executable @block_successors_ex_0 {
  flow.dispatch.entry @entry_0
  module {
    func @entry_0(%arg0: i32, %arg1: i32) -> i32 {
      %c0 = constant 0 : i32
      %c1 = constant 1 : i32
      %eqz = cmpi eq, %arg0, %arg1 : i32
      cond_br %eqz, ^bb_a(%c0 : i32), ^bb_b(%c1 : i32)
    ^bb_a(%bb_a_arg0 : i32):
      return %bb_a_arg0 : i32
    ^bb_b(%bb_b_arg0 : i32):
      return %bb_b_arg0 : i32
    }
  }
}
// CHECK-LABEL: flow.executable @block_successors_ex_with_swapped_cond_br
flow.executable @block_successors_ex_with_swapped_cond_br {
  flow.dispatch.entry @entry_1
  module {
    func @entry_0(%arg0: i32, %arg1: i32) -> i32 {
      %c0 = constant 0 : i32
      %c1 = constant 1 : i32
      %eqz = cmpi eq, %arg0, %arg1 : i32
      cond_br %eqz, ^bb_b(%c0 : i32), ^bb_b(%c1 : i32)
    ^bb_a(%bb_a_arg0 : i32):
      return %bb_a_arg0 : i32
    ^bb_b(%bb_b_arg0 : i32):
      return %bb_b_arg0 : i32
    }
  }
}
