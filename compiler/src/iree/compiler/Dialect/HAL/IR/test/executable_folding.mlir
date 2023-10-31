// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// Tests that multiple constant blocks get merged into one.

// CHECK-LABEL: @multiple_constant_blocks
hal.executable @multiple_constant_blocks {
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    // CHECK: hal.executable.constant.block() -> (i32, i32, i32) as ("foo", "bar", "baz")
    hal.executable.constant.block() -> (i32, i32) as ("foo", "bar") {
      // CHECK-DAG: %[[FOO:.+]] = arith.constant 0
      %foo = arith.constant 0 : i32
      // CHECK-DAG: %[[BAR:.+]] = arith.constant 1
      %bar = arith.constant 1 : i32
      // CHECK-DAG: %[[BAZ:.+]] = arith.constant 2
      // CHECK: hal.return %[[FOO]], %[[BAR]], %[[BAZ]] : i32, i32, i32
      hal.return %foo, %bar : i32, i32
    }
    // CHECK-NOT: hal.executable.constant.block
    hal.executable.constant.block() -> i32 as "baz" {
      %baz = arith.constant 2 : i32
      hal.return %baz : i32
    }
  }
}

// -----

// Tests that %device arguments and inner branches are handled when merging
// constant blocks. Various CFG canonicalizers run here and kind of make things
// weird but the key is that we end up with a single block with a single return
// that yields the expected values.

// CHECK-LABEL: @complex_constant_blocks
hal.executable @complex_constant_blocks {
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    // CHECK: hal.executable.constant.block(%[[DEVICE:.+]]: !hal.device) -> (i32, i32, i32) as ("foo", "bar", "baz")
    hal.executable.constant.block(%device: !hal.device) -> (i32, i32) as ("foo", "bar") {
      // CHECK-DAG: %[[DUMMY:.+]] = arith.constant 0
      %dummy = arith.constant 0 : i32
      // CHECK: %[[OK0:.+]], %[[QUERY0:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("sys" :: "foo")
      %ok0, %query0 = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i32
      // CHECK: cf.cond_br %[[OK0]], ^[[BB_OK0:[a-z0-9]+]], ^[[BB_CONT:[a-z0-9]+]](%[[DUMMY]], %[[DUMMY]] : i32, i32)
      cf.cond_br %ok0, ^bb_ok0, ^bb_fail
    // CHECK: ^[[BB_OK0]]:
    ^bb_ok0:
      // CHECK: %[[OK1:.+]], %[[QUERY1:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("sys" :: "bar")
      %ok1, %query1 = hal.device.query<%device : !hal.device> key("sys" :: "bar") : i1, i32
      // CHECK: cf.cond_br %[[OK1]], ^[[BB_CONT]](%[[QUERY0]], %[[QUERY1]] : i32, i32), ^[[BB_CONT]](%[[DUMMY]], %[[DUMMY]] : i32, i32)
      cf.cond_br %ok1, ^bb_ok1, ^bb_fail
    // Note that these blocks fold but we want them here so that we test the
    // multiple return -> single return path.
    ^bb_ok1:
      hal.return %query0, %query1 : i32, i32
    ^bb_fail:
      hal.return %dummy, %dummy : i32, i32

    // Second block gets spliced in here:
    // CHECK: ^[[BB_CONT]](%[[BB_QUERY0:.+]]: i32, %[[BB_QUERY1:.+]]: i32):
      // CHECK: %[[OK2:.+]], %[[QUERY2:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("sys" :: "baz")
      // CHECK: hal.return %[[BB_QUERY0]], %[[BB_QUERY1]], %[[QUERY2]]
    }
    // CHECK-NOT: hal.executable.constant.block
    hal.executable.constant.block(%device: !hal.device) -> i32 as "baz" {
      %ok2, %query2 = hal.device.query<%device : !hal.device> key("sys" :: "baz") : i1, i32
      hal.return %query2 : i32
    }
  }
}

// -----

// Tests that unused %device args are dropped. This isn't structurally critical
// but does make the IR easier to read after propagation may elide device
// queries.

// CHECK-LABEL: @unused_device_arg
hal.executable @unused_device_arg {
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    // CHECK: hal.executable.constant.block() -> i32 as "foo"
    hal.executable.constant.block(%device: !hal.device) -> i32 as "foo" {
      %c0 = arith.constant 0 : i32
      hal.return %c0 : i32
    }
  }
}

// -----

// Tests that keys get deduplicated in blocks such that only one value is
// produced. Duplication can happen after merging.

// CHECK-LABEL: @duplicate_keys
hal.executable @duplicate_keys {
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    // CHECK: hal.executable.constant.block() -> (i32, i32) as ("foo", "bar")
    hal.executable.constant.block() -> (i32, i32, i32) as ("foo", "bar", "foo") {
      // CHECK-DAG: %[[FOO:.+]] = arith.constant 1000
      %foo = arith.constant 1000 : i32
      // CHECK-DAG: %[[BAR:.+]] = arith.constant 2000
      %bar = arith.constant 2000 : i32
      // CHECK-NOT: arith.constant 1001
      %foo_dupe = arith.constant 1001 : i32
      // CHECK: hal.return %[[FOO]], %[[BAR]] : i32, i32
      hal.return %foo, %bar, %foo_dupe : i32, i32, i32
    }
  }
}

// -----

// Tests that duplicate keys get folded after merging constant blocks.
// This is just to test the combined interaction between merging and deduping.
// Note that "bar"

// CHECK-LABEL: @multiple_blocks_duplicate_keys
hal.executable @multiple_blocks_duplicate_keys {
  hal.executable.variant @backend target(#hal.executable.target<"backend", "format">) {
    // CHECK:  hal.executable.constant.block() -> (i32, i32) as ("foo", "bar")
    hal.executable.constant.block() -> (i32, i32) as ("foo", "bar") {
      // CHECK-DAG: %[[FOO:.+]] = arith.constant 0
      %foo = arith.constant 0 : i32
      // CHECK-DAG: %[[BAR:.+]] = arith.constant 1000
      %bar = arith.constant 1000 : i32
      // CHECK-NOT: arith.constant 1001
      // CHECK: hal.return %[[FOO]], %[[BAR]] : i32, i32
      hal.return %foo, %bar : i32, i32
    }
    // CHECK-NOT: hal.executable.constant.block
    hal.executable.constant.block() -> i32 as "bar" {
      %bar = arith.constant 1001 : i32
      hal.return %bar : i32
    }
  }
}
