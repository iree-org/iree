// RUN: iree-opt --split-input-file %s | FileCheck %s

#executable_target_format = #hal.executable.target<"backend", "format">

// CHECK-LABEL: @ex
hal.executable @ex {
  // CHECK: hal.executable.variant public @backend
  // CHECK-SAME: target(#executable_target_format)
  // CHECK-SAME: objects([#hal.executable.object<{path = "foo.bin"}>, #hal.executable.object<{path = "bar.bin"}>])
  hal.executable.variant @backend target(#executable_target_format) objects([
    #hal.executable.object<{path = "foo.bin"}>,
    #hal.executable.object<{path = "bar.bin"}>
  ]) {
    // CHECK-DAG: hal.executable.export public @entry0 ordinal(0) layout(#pipeline_layout)
    // CHECK:     workgroup_size = [4 : index, 1 : index, 1 : index]
    hal.executable.export public @entry0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>,
      #hal.pipeline.binding<storage_buffer>
    ]>) attributes {
      workgroup_size = [4 : index, 1 : index, 1 : index]
    }
  }
  // CHECK: hal.executable.binary
  hal.executable.binary @backend_binary attributes {
    // CHECK-SAME: data = dense<1> : vector<128xi8>,
    data = dense<1> : vector<128xi8>,
    // CHECK-SAME: format = "some_format"
    format = "some_format"
  }
}

// -----

#executable_target_format = #hal.executable.target<"backend", "format">

// CHECK-LABEL: @ex_with_workgroup_count_region
hal.executable @ex_with_workgroup_count_region {
  // CHECK: hal.executable.variant public @backend target(#executable_target_format
  hal.executable.variant @backend target(#executable_target_format) {
    // CHECK-DAG: hal.executable.export public @entry0 ordinal(0) layout(#pipeline_layout)
    // CHECK:     subgroup_size = 64 : index
    // CHECK:     workgroup_size = [4 : index, 1 : index, 1 : index]
    hal.executable.export public @entry0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>,
      #hal.pipeline.binding<storage_buffer>
    ]>) count(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      hal.return %arg0, %arg1, %arg2 : index, index, index
    } attributes {
      subgroup_size = 64 : index,
      workgroup_size = [4 : index, 1 : index, 1 : index]
    }
  }
  // CHECK: hal.executable.binary
  hal.executable.binary @backend_binary attributes {
    // CHECK-SAME: data = dense<1> : vector<128xi8>,
    data = dense<1> : vector<128xi8>,
    // CHECK-SAME: format = "some_format"
    format = "some_format"
  }
}

// -----

#executable_target_format = #hal.executable.target<"backend", "format">

// CHECK-LABEL: @ex_with_condition
hal.executable @ex_with_condition {
  // CHECK: hal.executable.variant public @backend target(#executable_target_format
  hal.executable.variant @backend target(#executable_target_format) {
    // CHECK: hal.executable.condition(%[[DEVICE:.+]]: !hal.device) -> i1 {
    hal.executable.condition(%device: !hal.device) -> i1 {
      // CHECK-NEXT: %[[OK:.+]], %[[VALUE:.+]] = hal.device.query<%[[DEVICE]]
      %ok, %value = hal.device.query<%device : !hal.device> key("some" :: "value") : i1, i32
      // CHECK-NEXT: hal.return %[[OK]]
      hal.return %ok : i1
    }

    // CHECK-DAG: hal.executable.export public @entry0 ordinal(0) layout(#pipeline_layout)
    // CHECK:     subgroup_size = 64 : index
    // CHECK:     workgroup_size = [4 : index, 1 : index, 1 : index]
    hal.executable.export public @entry0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>,
      #hal.pipeline.binding<storage_buffer>
    ]>) count(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      hal.return %arg0, %arg1, %arg2 : index, index, index
    } attributes {
      subgroup_size = 64 : index,
      workgroup_size = [4 : index, 1 : index, 1 : index]
    }
  }
  // CHECK: hal.executable.binary
  hal.executable.binary @backend_binary attributes {
    // CHECK-SAME: data = dense<1> : vector<128xi8>,
    data = dense<1> : vector<128xi8>,
    // CHECK-SAME: format = "some_format"
    format = "some_format"
  }
}

// -----

#executable_target_format = #hal.executable.target<"backend", "format">

// CHECK-LABEL: @ex_with_constants
hal.executable @ex_with_constants {
  // CHECK: hal.executable.variant public @backend
  hal.executable.variant @backend target(#executable_target_format) {
    // CHECK: hal.executable.constant.block(%{{.+}}: !hal.device) -> (i32, i32) as ("foo", "bar")
    hal.executable.constant.block(%device: !hal.device) -> (i32, i32) as ("foo", "bar") {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      hal.return %c0, %c1 : i32, i32
    }
    // CHECK: hal.executable.constant.block(%{{.+}}: !hal.device) -> i32 as "baz"
    hal.executable.constant.block(%device: !hal.device) -> i32 as "baz" {
      %c2 = arith.constant 2 : i32
      hal.return %c2 : i32
    }
    builtin.module {
      func.func @dispatch0() {
        // CHECK: = hal.executable.constant.load "foo" : i32
        %0 = hal.executable.constant.load "foo" : i32
        // CHECK: = hal.executable.constant.load "bar" : i32
        %1 = hal.executable.constant.load "bar" : i32
        // CHECK: = hal.executable.constant.load "baz" : i32
        %2 = hal.executable.constant.load "baz" : i32
        func.return
      }
    }
  }
}

// -----

// CHECK-LABEL: @executable_create
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device
// CHECK-SAME: %[[AFFINITY:.+]]: i64
util.func public @executable_create(%device: !hal.device, %affinity: i64) {
  //      CHECK: = hal.executable.create
  // CHECK-SAME:     device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:   affinity(%[[AFFINITY]])
  // CHECK-SAME:     target(@exe::@binary1) : !hal.executable
  %0 = hal.executable.create device(%device : !hal.device)
                           affinity(%affinity)
                             target(@exe::@binary1) : !hal.executable
  util.return
}

// -----

// CHECK-LABEL: @unresolved_workload_ex
hal.executable @unresolved_workload_ex {
  // CHECK: hal.executable.variant public @backend
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    // CHECK: hal.executable.export public @entry0
    hal.executable.export public @entry0 ordinal(0) layout(#hal.pipeline.layout<bindings = [
      #hal.pipeline.binding<storage_buffer>,
      #hal.pipeline.binding<storage_buffer>
    ]>) count(%device: !hal.device, %arg0: index) -> (index, index, index) {
      hal.return %arg0, %arg0, %arg0 : index, index, index
    }
  }
}
// CHECK-LABEL: @unresolved_workload
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:  %[[WORKLOAD_0:.+]]: index, %[[WORKLOAD_1:.+]]: index)
util.func public @unresolved_workload(
    %device: !hal.device,
    %workload_0: index, %workload_1: index) -> (index, index, index) {
  // CHECK: %[[WORKGROUP_X:.+]], %[[WORKGROUP_Y:.+]], %[[WORKGROUP_Z:.+]] =
  // CHECK-SAME:   hal.executable.calculate_workgroups
  // CHECK-SAME:       device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:       target(@unresolved_workload_ex::@backend::@entry0)
  // CHECK-SAME:       workload([%[[WORKLOAD_0]], %[[WORKLOAD_1]]]) : index, index, index
  %workgroups:3 = hal.executable.calculate_workgroups
      device(%device : !hal.device)
      target(@unresolved_workload_ex::@backend::@entry0)
      workload([%workload_0, %workload_1]) : index, index, index
  // CHECK: util.return %[[WORKGROUP_X]], %[[WORKGROUP_Y]], %[[WORKGROUP_Z]]
  util.return %workgroups#0, %workgroups#1, %workgroups#2 : index, index, index
}
