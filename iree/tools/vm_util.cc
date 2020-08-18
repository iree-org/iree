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
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "iree/base/file_io.h"
#include "iree/base/signature_mangle.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/bytecode_module.h"

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

StatusOr<vm::ref<iree_vm_list_t>> ParseToVariantList(
    absl::Span<const RawSignatureParser::Description> descs,
    iree_hal_allocator_t* allocator,
    absl::Span<const absl::string_view> input_strings) {
  if (input_strings.size() != descs.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Signature mismatch; expected " << descs.size()
           << " buffer strings but received " << input_strings.size();
  }
  vm::ref<iree_vm_list_t> variant_list;
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(/*element_type=*/nullptr, input_strings.size(),
                          iree_allocator_system(), &variant_list));
  for (size_t i = 0; i < input_strings.size(); ++i) {
    auto input_string = input_strings[i];
    auto desc = descs[i];
    std::string desc_str;
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
        iree_vm_value_t val = iree_vm_value_make_i32(0);
        if (!absl::SimpleAtoi(input_view, &val.i32)) {
          return InvalidArgumentErrorBuilder(IREE_LOC)
                 << "Converting '" << input_view << "' to i32 when parsing '"
                 << input_string << "'";
        }
        IREE_RETURN_IF_ERROR(iree_vm_list_push_value(variant_list.get(), &val));
        break;
      }
      case RawSignatureParser::Type::kBuffer: {
        iree_hal_buffer_view_t* buffer_view = nullptr;
        IREE_RETURN_IF_ERROR(iree_hal_buffer_view_parse(
            iree_string_view_t{input_string.data(), input_string.size()},
            allocator, iree_allocator_system(), &buffer_view))
            << "Parsing value '" << input_string << "'";
        auto buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
        IREE_RETURN_IF_ERROR(
            iree_vm_list_push_ref_move(variant_list.get(), &buffer_view_ref));
        break;
      }
      default:
        return UnimplementedErrorBuilder(IREE_LOC)
               << "Unsupported signature type: " << desc_str;
    }
  }
  return variant_list.release();
}

StatusOr<vm::ref<iree_vm_list_t>> ParseToVariantList(
    absl::Span<const RawSignatureParser::Description> descs,
    iree_hal_allocator_t* allocator,
    absl::Span<const std::string> input_strings) {
  absl::InlinedVector<absl::string_view, 4> input_views(input_strings.size());
  for (int i = 0; i < input_strings.size(); ++i) {
    input_views[i] = input_strings[i];
  }
  return ParseToVariantList(descs, allocator, input_views);
}

StatusOr<vm::ref<iree_vm_list_t>> ParseToVariantListFromFile(
    absl::Span<const RawSignatureParser::Description> descs,
    iree_hal_allocator_t* allocator, const std::string& filename) {
  IREE_ASSIGN_OR_RETURN(auto file_string, file_io::GetFileContents(filename));
  absl::InlinedVector<absl::string_view, 4> input_views(
      absl::StrSplit(file_string, '\n', absl::SkipEmpty()));
  return ParseToVariantList(descs, allocator, input_views);
}

Status PrintVariantList(absl::Span<const RawSignatureParser::Description> descs,
                        iree_vm_list_t* variant_list, std::ostream* os) {
  for (int i = 0; i < iree_vm_list_size(variant_list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_variant(variant_list, i, &variant))
        << "variant " << i << "not present";

    const auto& desc = descs[i];
    std::string desc_str;
    desc.ToString(desc_str);
    LOG(INFO) << "result[" << i << "]: " << desc_str;

    switch (desc.type) {
      case RawSignatureParser::Type::kScalar: {
        if (variant.type.value_type != IREE_VM_VALUE_TYPE_I32) {
          return InvalidArgumentErrorBuilder(IREE_LOC)
                 << "variant " << i << " has value type "
                 << static_cast<int>(variant.type.value_type)
                 << " but descriptor information " << desc_str;
        }
        if (desc.scalar.type != AbiConstants::ScalarType::kSint32) {
          return UnimplementedErrorBuilder(IREE_LOC)
                 << "Unsupported signature scalar type: " << desc_str;
        }
        *os << "i32=" << variant.i32 << "\n";
        break;
      }
      case RawSignatureParser::Type::kBuffer: {
        if (!iree_vm_type_def_is_ref(&variant.type)) {
          return InvalidArgumentErrorBuilder(IREE_LOC)
                 << "variant " << i << " has value type "
                 << static_cast<int>(variant.type.value_type)
                 << " but descriptor information " << desc_str;
        }
        auto* buffer_view = iree_hal_buffer_view_deref(&variant.ref);
        if (!buffer_view) {
          return InvalidArgumentErrorBuilder(IREE_LOC)
                 << "failed dereferencing variant " << i;
        }

        std::string result_str(4096, '\0');
        iree_status_t status;
        do {
          iree_host_size_t actual_length = 0;
          status = iree_hal_buffer_view_format(
              buffer_view, /*max_element_count=*/1024, result_str.size() + 1,
              &result_str[0], &actual_length);
          result_str.resize(actual_length);
        } while (iree_status_is_out_of_range(status));
        IREE_RETURN_IF_ERROR(status);

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
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_create_driver(
      iree_string_view_t{driver_name.data(), driver_name.size()},
      iree_allocator_system(), &driver))
      << "Creating driver '" << driver_name << "'";
  IREE_RETURN_IF_ERROR(iree_hal_driver_create_default_device(
      driver, iree_allocator_system(), out_device))
      << "Creating default device for driver '" << driver_name << "'";
  iree_hal_driver_release(driver);
  return OkStatus();
}

Status CreateHalModule(iree_hal_device_t* device,
                       iree_vm_module_t** out_module) {
  IREE_RETURN_IF_ERROR(
      iree_hal_module_create(device, iree_allocator_system(), out_module))
      << "Creating HAL module";
  return OkStatus();
}

Status LoadBytecodeModule(absl::string_view module_data,
                          iree_vm_module_t** out_module) {
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_data.data()),
          module_data.size()},
      iree_allocator_null(), iree_allocator_system(), out_module))
      << "Deserializing module";
  return OkStatus();
}
}  // namespace iree
