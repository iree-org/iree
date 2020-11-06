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
  // CHECK: %0 = flow.dispatch @single_executable_ex_0::@single_executable_entry_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @single_executable_ex_0::@single_executable_entry_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
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
  // CHECK: %0 = flow.dispatch @duplicate_executables_ex_0::@duplicate_executables_entry_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @duplicate_executables_ex_0::@duplicate_executables_entry_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %1 = flow.dispatch @duplicate_executables_ex_0::@duplicate_executables_entry_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = flow.dispatch @duplicate_executables_ex_1::@duplicate_executables_entry_1[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %2 = flow.dispatch @duplicate_executables_ex_2::@duplicate_executables_entry_2[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = flow.dispatch @duplicate_executables_ex_2::@duplicate_executables_entry_2[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
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
  // CHECK: %0 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %1 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_1[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_1[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %2 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = flow.dispatch @multiple_entry_points_ex_1::@multiple_entry_points_1_entry_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %3 = flow.dispatch @multiple_entry_points_ex_0::@multiple_entry_points_0_entry_1[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = flow.dispatch @multiple_entry_points_ex_1::@multiple_entry_points_1_entry_1[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
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
  // CHECK: %0 = flow.dispatch @different_types_float_ex::@different_types_float_entry[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  %0 = flow.dispatch @different_types_float_ex::@different_types_float_entry[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  // CHECK: %1 = flow.dispatch @different_types_int_ex::@different_types_int_entry[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  %1 = flow.dispatch @different_types_int_ex::@different_types_int_entry[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
