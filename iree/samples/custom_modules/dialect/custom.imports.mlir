// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Describes a custom module implemented in native code.
// The imports are used for mapping higher-level IR types to the VM ABI and for
// attaching additional attributes for compiler optimization.
//
// Each import defined here has a matching function exported from the native
// module (custom_modules/native_module.cc). In most cases an op in the source
// dialect will map directly to an import here, though it's possible for
// versioning and overrides to cause M:N mappings.
vm.module @custom {

// Formats the tensor using the IREE buffer printer to have a shape/type and
// the contents as a string.
vm.import @buffer_to_message(
  %buffer_view : !iree.ref<!hal.buffer_view>
) -> !iree.ref<!custom.message>
attributes {nosideeffects}

// Parses the message containing a IREE buffer parser-formatted tensor.
vm.import @message_to_buffer(
  %message : !iree.ref<!custom.message>
) -> !iree.ref<!hal.buffer_view>
attributes {nosideeffects}

// Prints the %message provided %count times.
// Maps to the IREE::Custom::PrintOp.
vm.import @print(
  %message : !iree.ref<!custom.message>,
  %count : i32
)

// Returns the message with its characters reversed.
// Maps to the IREE::Custom::ReverseOp.
vm.import @reverse(
  %message : !iree.ref<!custom.message>
) -> !iree.ref<!custom.message>
attributes {nosideeffects}

// Returns a per-context unique message.
// Maps to the IREE::Custom::GetUniqueMessageOp.
vm.import @get_unique_message() -> !iree.ref<!custom.message>
attributes {nosideeffects}

}  // vm.module
