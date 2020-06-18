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

#include "iree/tools/vm_util.h"

#include <ostream>

#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "iree/base/api_util.h"
#include "iree/base/buffer_string_util.h"
#include "iree/base/shape.h"
#include "iree/base/shaped_buffer.h"
#include "iree/base/shaped_buffer_string_util.h"
#include "iree/base/signature_mangle.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/module.h"
#include "iree/vm/variant_list.h"

namespace iree {

Status ValidateFunctionAbi(const iree_vm_function_t& function) {
  iree_string_view_t sig_fv =
      iree_vm_function_reflection_attr(&function, iree_make_cstring_view("fv"));
  if (absl::string_view{sig_fv.data, sig_fv.size} != "1") {
    auto function_name = iree_vm_function_name(&function);
    return iree::UnimplementedErrorBuilder(IREE_LOC)
           << "Unsupported function ABI for: '"
           << absl::string_view(function_name.data, function_name.size) << "'("
           << absl::string_view{sig_fv.data, sig_fv.size} << ")";
  }
  return OkStatus();
}

StatusOr<std::vector<RawSignatureParser::Description>> ParseInputSignature(
    iree_vm_function_t& function) {
  iree_string_view_t sig_f =
      iree_vm_function_reflection_attr(&function, iree_make_cstring_view("f"));
  RawSignatureParser sig_parser;
  std::vector<RawSignatureParser::Description> input_descs;
  sig_parser.VisitInputs(absl::string_view{sig_f.data, sig_f.size},
                         [&](const RawSignatureParser::Description& desc) {
                           input_descs.push_back(desc);
                         });
  if (sig_parser.GetError()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Parsing function signature '" << sig_f.data
           << "' failed getting input";
  }
  return input_descs;
}

StatusOr<std::vector<RawSignatureParser::Description>> ParseOutputSignature(
    const iree_vm_function_t& function) {
  iree_string_view_t sig_f =
      iree_vm_function_reflection_attr(&function, iree_make_cstring_view("f"));
  RawSignatureParser sig_parser;
  std::vector<RawSignatureParser::Description> output_descs;
  sig_parser.VisitResults(absl::string_view{sig_f.data, sig_f.size},
                          [&](const RawSignatureParser::Description& desc) {
                            output_descs.push_back(desc);
                          });
  if (sig_parser.GetError()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Parsing function signature '" << sig_f.data
           << "' failed getting results";
  }
  return output_descs;
}

StatusOr<iree_vm_variant_list_t*> ParseToVariantList(
    absl::Span<const RawSignatureParser::Description> descs,
    iree_hal_allocator_t* allocator,
    absl::Span<const std::string> input_strings) {
  if (input_strings.size() != descs.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Signature mismatch; expected " << descs.size()
           << " buffer strings but received " << input_strings.size();
  }
  iree_vm_variant_list_t* variant_list = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_vm_variant_list_alloc(input_strings.size(), IREE_ALLOCATOR_SYSTEM,
                                 &variant_list),
      IREE_LOC));
  for (int i = 0; i < input_strings.size(); ++i) {
    auto input_string = input_strings[i];
    auto desc = descs[i];
    std::string desc_str;
    // TODO(laurenzo): Parse real element type once bindings code enforces it.
    iree_hal_element_type_t hal_element_type = IREE_HAL_ELEMENT_TYPE_NONE;
    desc.ToString(desc_str);
    switch (desc.type) {
      case RawSignatureParser::Type::kScalar: {
        if (desc.scalar.type != AbiConstants::ScalarType::kSint32) {
          return UnimplementedErrorBuilder(IREE_LOC)
                 << "Unsupported signature scalar type: " << desc_str;
        }
        absl::string_view input_view = absl::StripAsciiWhitespace(input_string);
        input_view = absl::StripPrefix(input_view, "\"");
        input_view = absl::StripSuffix(input_view, "\"");
        if (!absl::ConsumePrefix(&input_view, "i32=")) {
          return InvalidArgumentErrorBuilder(IREE_LOC)
                 << "Parsing '" << input_string
                 << "'. Has i32 descriptor but does not start with 'i32='";
        }
        int32_t val;
        if (!absl::SimpleAtoi(input_view, &val)) {
          return InvalidArgumentErrorBuilder(IREE_LOC)
                 << "Converting '" << input_view << "' to i32 when parsing '"
                 << input_string << "'";
        }
        iree_vm_variant_list_append_value(variant_list,
                                          iree_vm_value_make_i32(val));
        break;
      }
      case RawSignatureParser::Type::kBuffer: {
        ASSIGN_OR_RETURN(auto shaped_buffer,
                         ParseShapedBufferFromString(input_string),
                         _ << "Parsing value '" << input_string << "'");
        // Allocate the buffer.
        iree_hal_buffer_t* buf = nullptr;
        // TODO(benvanik): combined function for linear to optimal upload.
        iree_device_size_t allocation_size =
            shaped_buffer.shape().element_count() *
            shaped_buffer.element_size();
        RETURN_IF_ERROR(FromApiStatus(
            iree_hal_allocator_allocate_buffer(
                allocator,
                static_cast<iree_hal_memory_type_t>(
                    IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
                static_cast<iree_hal_buffer_usage_t>(
                    IREE_HAL_BUFFER_USAGE_ALL | IREE_HAL_BUFFER_USAGE_CONSTANT),
                allocation_size, &buf),
            IREE_LOC))
            << "Allocating buffer";
        RETURN_IF_ERROR(FromApiStatus(
            iree_hal_buffer_write_data(buf, 0, shaped_buffer.contents().data(),
                                       shaped_buffer.contents().size()),
            IREE_LOC))
            << "Populating buffer contents ";

        absl::InlinedVector<iree_hal_dim_t, 5> dims(
            shaped_buffer.shape().size());
        // TODO(laurenzo): The following should work but Shape iterators
        // cause access violations.
        // std::copy(shaped_buffer.shape().begin(), shaped_buffer.shape().end(),
        //           dims.begin());
        for (size_t i = 0; i < dims.size(); ++i) {
          dims[i] = shaped_buffer.shape()[i];
        }

        // Wrap in buffer view.
        iree_hal_buffer_view_t* buffer_view = nullptr;
        RETURN_IF_ERROR(FromApiStatus(
            iree_hal_buffer_view_create(buf, dims.data(), dims.size(),
                                        hal_element_type, IREE_ALLOCATOR_SYSTEM,
                                        &buffer_view),
            IREE_LOC))
            << "Creating buffer view";
        iree_hal_buffer_release(buf);
        auto buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
        RETURN_IF_ERROR(FromApiStatus(iree_vm_variant_list_append_ref_move(
                                          variant_list, &buffer_view_ref),
                                      IREE_LOC));
        break;
      }
      default:
        return UnimplementedErrorBuilder(IREE_LOC)
               << "Unsupported signature type: " << desc_str;
    }
  }
  return variant_list;
}

Status PrintVariantList(absl::Span<const RawSignatureParser::Description> descs,
                        iree_vm_variant_list_t* variant_list,
                        std::ostream* os) {
  for (int i = 0; i < iree_vm_variant_list_size(variant_list); ++i) {
    iree_vm_variant_t* variant = iree_vm_variant_list_get(variant_list, i);
    if (!variant) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "variant " << i << "not present";
    }

    const auto& desc = descs[i];
    std::string desc_str;
    desc.ToString(desc_str);
    LOG(INFO) << "result[" << i << "]: " << desc_str;

    switch (desc.type) {
      case RawSignatureParser::Type::kScalar: {
        if (variant->value_type != IREE_VM_VALUE_TYPE_I32) {
          return InvalidArgumentErrorBuilder(IREE_LOC)
                 << "variant " << i << " has value type "
                 << static_cast<int>(variant->value_type)
                 << " but descriptor information " << desc_str;
        }
        if (desc.scalar.type != AbiConstants::ScalarType::kSint32) {
          return UnimplementedErrorBuilder(IREE_LOC)
                 << "Unsupported signature scalar type: " << desc_str;
        }
        *os << "i32=" << variant->i32 << "\n";
        break;
      }
      case RawSignatureParser::Type::kBuffer: {
        if (variant->value_type != IREE_VM_VALUE_TYPE_NONE) {
          return InvalidArgumentErrorBuilder(IREE_LOC)
                 << "variant " << i << " has value type "
                 << static_cast<int>(variant->value_type)
                 << " but descriptor information " << desc_str;
        }
        auto* buffer_view = iree_hal_buffer_view_deref(&variant->ref);
        if (!buffer_view) {
          return InvalidArgumentErrorBuilder(IREE_LOC)
                 << "failed dereferencing variant " << i;
        }
        auto* buffer = iree_hal_buffer_view_buffer(buffer_view);
        if (!buffer) {
          return InvalidArgumentErrorBuilder(IREE_LOC)
                 << "failed to get buffer for variant " << i;
        }

        auto print_mode = BufferDataPrintMode::kFloatingPoint;
        int8_t element_size = 4;

        // Copy the dims out of the buffer view.
        absl::InlinedVector<iree_hal_dim_t, 5> dims32(
            iree_hal_buffer_view_shape_rank(buffer_view));
        iree_hal_buffer_view_shape(buffer_view, dims32.size(), dims32.data(),
                                   nullptr);
        absl::InlinedVector<int, 5> dims(dims32.size());
        std::copy(dims32.begin(), dims32.end(), dims.begin());
        Shape shape{dims};

        switch (desc.buffer.scalar_type) {
          case AbiConstants::ScalarType::kIeeeFloat16:
          case AbiConstants::ScalarType::kIeeeFloat32:
          case AbiConstants::ScalarType::kIeeeFloat64:
            print_mode = BufferDataPrintMode::kFloatingPoint;
            break;
          case AbiConstants::ScalarType::kSint8:
          case AbiConstants::ScalarType::kSint16:
          case AbiConstants::ScalarType::kSint32:
          case AbiConstants::ScalarType::kSint64:
            print_mode = BufferDataPrintMode::kSignedInteger;
            break;
          case AbiConstants::ScalarType::kUint8:
          case AbiConstants::ScalarType::kUint16:
          case AbiConstants::ScalarType::kUint32:
          case AbiConstants::ScalarType::kUint64:
            print_mode = BufferDataPrintMode::kUnsignedInteger;
            break;
          default:
            print_mode = BufferDataPrintMode::kBinary;
            break;
        }
        element_size = AbiConstants::kScalarTypeSize[static_cast<unsigned>(
            desc.buffer.scalar_type)];

        iree_hal_mapped_memory_t mapped_memory;
        RETURN_IF_ERROR(FromApiStatus(
            iree_hal_buffer_map(buffer, IREE_HAL_MEMORY_ACCESS_READ, 0,
                                IREE_WHOLE_BUFFER, &mapped_memory),
            IREE_LOC))
            << "mapping hal buffer";
        auto contents = absl::MakeConstSpan(mapped_memory.contents.data,
                                            mapped_memory.contents.data_length);
        ShapedBuffer shaped_buffer(
            element_size, shape,
            std::vector<uint8_t>(contents.begin(), contents.end()));
        ASSIGN_OR_RETURN(auto result_str, PrintShapedBufferToString(
                                              shaped_buffer, print_mode, 1024));
        iree_hal_buffer_unmap(buffer, &mapped_memory);
        *os << result_str << "\n";
        break;
      }
      default:
        return UnimplementedErrorBuilder(IREE_LOC)
               << "Unsupported signature type: " << desc_str;
    }
  }

  return OkStatus();
}

Status CreateDevice(absl::string_view driver_name,
                    iree_hal_device_t** out_device) {
  LOG(INFO) << "Creating driver and device for '" << driver_name << "'...";
  iree_hal_driver_t* driver = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_driver_registry_create_driver(
          iree_string_view_t{driver_name.data(), driver_name.size()},
          IREE_ALLOCATOR_SYSTEM, &driver),
      IREE_LOC))
      << "Creating driver '" << driver_name << "'";
  RETURN_IF_ERROR(FromApiStatus(iree_hal_driver_create_default_device(
                                    driver, IREE_ALLOCATOR_SYSTEM, out_device),
                                IREE_LOC))
      << "Creating default device for driver '" << driver_name << "'";
  iree_hal_driver_release(driver);
  return OkStatus();
}

Status CreateHalModule(iree_hal_device_t* device,
                       iree_vm_module_t** out_module) {
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_module_create(device, IREE_ALLOCATOR_SYSTEM, out_module),
      IREE_LOC))
      << "Creating HAL module";
  return OkStatus();
}

Status LoadBytecodeModule(absl::string_view module_data,
                          iree_vm_module_t** out_module) {
  RETURN_IF_ERROR(FromApiStatus(
      iree_vm_bytecode_module_create(
          iree_const_byte_span_t{
              reinterpret_cast<const uint8_t*>(module_data.data()),
              module_data.size()},
          IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_SYSTEM, out_module),
      IREE_LOC))
      << "Deserializing module";
  return OkStatus();
}
}  // namespace iree
