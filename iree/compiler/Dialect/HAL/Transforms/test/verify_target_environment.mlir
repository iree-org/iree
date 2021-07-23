// RUN: iree-opt -split-input-file -iree-hal-verify-target-environment %s -verify-diagnostics | IreeFileCheck %s

// expected-error@+1 {{no HAL target devices specified}}
module @module {}

// -----

// expected-error@+1 {{no HAL target devices specified}}
module @module attributes {hal.device.targets = []} {}

// -----

// expected-error@+1 {{invalid target attr type}}
module @module attributes {hal.device.targets = ["wrong_type"]} {}

// -----

// expected-error@+1 {{unregistered target backend "foo"}}
module @module attributes {hal.device.targets = [#hal.device.target<"foo">]} {}

// -----

#device_target_vmvx = #hal.device.target<"vmvx">

// CHECK: module @module attributes {hal.device.targets = [#device_target_vmvx]}
module @module attributes {hal.device.targets = [#device_target_vmvx]} {}
