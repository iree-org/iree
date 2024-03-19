// RUN: iree-opt --split-input-file --iree-hal-loader-materialize-executables %s | FileCheck %s

// Tests that executable binaries get moved to initialized globals and lookups
// get rewritten to point at the globals. Note that we do 2 to ensure we're
// enumerating all executables and all lookups.

// CHECK-LABEL: util.global private @ex0 : !hal.executable
// CHECK: util.initializer {
// CHECK:   %[[FORMAT_SUPPORTED:.+]] = hal_loader.executable.query_support format("embedded-elf-x86_64") : i1
// CHECK:   cf.cond_br %[[FORMAT_SUPPORTED]], ^bb2, ^bb3
// CHECK: ^bb2:
// CHECK:   %[[BINARY_DATA:.+]] = util.buffer.constant "binary" {alignment = 64 : index, mime_type = "application/x-elf"} : !util.buffer = dense<123> : vector<64xi8>
// CHECK:   %[[EXECUTABLE:.+]] = hal_loader.executable.load format("embedded-elf-x86_64") data(%[[BINARY_DATA]]) : !hal.executable
// CHECK:   cf.br ^bb4(%[[EXECUTABLE]] : !hal.executable)
// CHECK: ^bb3:
// CHECK:   util.status.check_ok
// CHECK:   util.return
// CHECK: ^bb4(%[[STORE_VALUE:.+]]: !hal.executable):
// CHECK:   util.global.store %[[STORE_VALUE]], @ex0 : !hal.executable
// CHECK:   util.return
// CHECK: }
hal.executable private @ex0 {
  hal.executable.binary public @binary attributes {data = dense<123> : vector<64xi8>, format = "embedded-elf-x86_64", mime_type = "application/x-elf"}
}

// CHECK-LABEL: @get_ex0
util.func private @get_ex0() -> !hal.executable {
  // CHECK: %[[EX0:.+]] = util.global.load @ex0 : !hal.executable
  %ex0 = hal_loader.executable.lookup executable(@ex0) : !hal.executable
  // CHECK: return %[[EX0]]
  util.return %ex0 : !hal.executable
}

// CHECK: util.global private @ex1 : !hal.executable
// CHECK: util.initializer
// CHECK:   hal_loader.executable.load format("embedded-elf-aarch64")
// CHECK:   util.global.store {{.+}}, @ex1
hal.executable private @ex1 {
  hal.executable.binary public @binary attributes {data = dense<123> : vector<64xi8>, format = "embedded-elf-aarch64", mime_type = "application/x-elf"}
}

// CHECK-LABEL: @get_ex1
util.func private @get_ex1() -> !hal.executable {
  // CHECK: %[[EX1:.+]] = util.global.load @ex1 : !hal.executable
  %ex1 = hal_loader.executable.lookup executable(@ex1) : !hal.executable
  // CHECK: return %[[EX1]]
  util.return %ex1 : !hal.executable
}
