// Copyright 2020 Google LLC
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

#ifndef IREE_TOOLS_VM_UTIL_H_
#define IREE_TOOLS_VM_UTIL_H_

#include <iostream>
#include <ostream>
#include <vector>

#include "absl/types/span.h"
#include "iree/base/signature_mangle.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "iree/vm/ref_cc.h"

namespace iree {

// TODO(benvanik) Update these when we can use RAII with the C API.

// Validates the ABI of the specified function is supported by current tooling.
Status ValidateFunctionAbi(const iree_vm_function_t& function);

// Returns descriptors for the input of the given function.
StatusOr<std::vector<RawSignatureParser::Description>> ParseInputSignature(
    iree_vm_function_t& function);

// Returns descriptors for the output of the given function.
StatusOr<std::vector<RawSignatureParser::Description>> ParseOutputSignature(
    const iree_vm_function_t& function);

// Parses |input_strings| into a variant list of VM scalars and buffers.
// Scalars should be in the format:
//   type=value
// Buffers should be in the IREE standard shaped buffer format:
//   [shape]xtype=[value]
// described in
// https://github.com/google/iree/tree/main/iree/hal/api.h
// Uses |allocator| to allocate the buffers.
// Uses descriptors in |descs| for type information and validation.
// The returned variant list must be freed by the caller.
StatusOr<vm::ref<iree_vm_list_t>> ParseToVariantList(
    absl::Span<const RawSignatureParser::Description> descs,
    iree_hal_allocator_t* allocator,
    absl::Span<const absl::string_view> input_strings);
StatusOr<vm::ref<iree_vm_list_t>> ParseToVariantList(
    absl::Span<const RawSignatureParser::Description> descs,
    iree_hal_allocator_t* allocator,
    absl::Span<const std::string> input_strings);

// Parses the content in |filename| into a variant list of VM scalars and
// buffers. See ParseToVariantList for the format of scalars and buffers. The
// inputs are expected to be newline-separated.
StatusOr<vm::ref<iree_vm_list_t>> ParseToVariantListFromFile(
    absl::Span<const RawSignatureParser::Description> descs,
    iree_hal_allocator_t* allocator, const std::string& filename);

// Prints a variant list of VM scalars and buffers to |os|.
// Prints scalars in the format:
//   type=value
// Prints buffers in the IREE standard shaped buffer format:
//   [shape]xtype=[value]
// described in
// https://github.com/google/iree/tree/main/iree/hal/api.h
// Uses descriptors in |descs| for type information and validation.
Status PrintVariantList(absl::Span<const RawSignatureParser::Description> descs,
                        iree_vm_list_t* variant_list,
                        std::ostream* os = &std::cout);

// Creates the default device for |driver| in |out_device|.
// The returned |out_device| must be released by the caller.
Status CreateDevice(absl::string_view driver_name,
                    iree_hal_device_t** out_device);

// Creates a hal module |driver| in |out_hal_module|.
// The returned |out_module| must be released by the caller.
Status CreateHalModule(iree_hal_device_t* device,
                       iree_vm_module_t** out_module);

// Loads a VM bytecode from an opaque string.
// The returned |out_module| must be released by the caller.
Status LoadBytecodeModule(absl::string_view module_data,
                          iree_vm_module_t** out_module);

}  // namespace iree

#endif  // IREE_TOOLS_VM_UTIL_H_
