// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Describes a custom module implemented in native code.
// The imports are used for mapping higher-level IR types to the VM ABI and for
// attaching additional attributes for compiler optimization.
//
// Each import defined here has a matching function exported from the native
// module (strings/native_module.cc). In most cases an op in the source
// dialect will map directly to an import here, though it's possible for
// versioning and overrides to cause M:N mappings.
vm.module @strings {

// Returns the string representation of an i32.
// Maps to the IREE::Strings::I32ToString.
vm.import @i32_to_string(%value : i32) -> !vm.ref<!strings.string>
attributes {nosideeffects}

// Elementwise conversion of a tensor of values to a tensor of strings.
// Maps to the IREE::Strings::ToStringTensor.
vm.import @to_string_tensor(%value : !vm.ref<!hal.buffer_view>) -> !vm.ref<!strings.string_tensor>
attributes {nosideeffects}

// Prints the contents of a string.
// Maps to the IREE::Strings::Print.
vm.import @print(%value : !vm.ref<!strings.string>)

// Converts the contents of a StringTensor to a String
// Maps to the IREE::Strings::StringTensortoString.
vm.import @string_tensor_to_string(%value : !vm.ref<!strings.string_tensor>) -> !vm.ref<!strings.string>

// Gathers all the strings from a Tensor by ID
// Maps to the IREE::Strings::Gather.
vm.import @gather(%value1 : !vm.ref<!strings.string_tensor>, %value2 : !vm.ref<!hal.buffer_view>) -> !vm.ref<!strings.string_tensor>

// Concatenates the strings in the tensor along the last dimension
// Maps to the IREE::Strings::Concat.
vm.import @concat(%value : !vm.ref<!strings.string_tensor>) -> !vm.ref<!strings.string_tensor>

}  // vm.module
