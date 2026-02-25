// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loads a VM module from the provided file and outputs the requested
// information.
//
// $ iree-dump-module [--output=...] module.vmfb
//   --output=metadata: module metadata and size breakdown.
//   --output=flatbuffer-binary: module flatbuffer in binary format.
//   --output=flatbuffer-json: module flatbuffer in JSON format.

#include <ctype.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/io/stdio_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/archive.h"
#include "iree/vm/bytecode/module.h"

// NOTE: include order matters:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/bytecode_module_def_json_printer.h"
#include "iree/schemas/bytecode_module_def_reader.h"
#include "iree/schemas/bytecode_module_def_verifier.h"

//===----------------------------------------------------------------------===//
// IO utilities
//===----------------------------------------------------------------------===//

static void iree_tooling_print_indent(int indent) {
  fprintf(stdout, "%*s", indent, "");
}

static void iree_tooling_printf_section_header(const char* format, ...) {
  fprintf(stdout,
          "//"
          "===-----------------------------------------------------------------"
          "---------------------------------------------===//\n");
  fprintf(stdout, "// ");
  va_list args;
  va_start(args, format);
  vfprintf(stdout, format, args);
  va_end(args);
  fprintf(stdout, "\n");
  fprintf(stdout,
          "//"
          "===-----------------------------------------------------------------"
          "---------------------------------------------===//\n");
}

//===----------------------------------------------------------------------===//
// --output=metadata
//===----------------------------------------------------------------------===//

static size_t iree_tooling_print_feature_bits(iree_vm_FeatureBits_enum_t bits) {
  size_t length = 0;
  if (bits & iree_vm_FeatureBits_EXT_F32) {
    if (length) fprintf(stdout, " | ");
    fprintf(stdout, "EXT_F32");
    length += strlen("EXT_F32");
    bits = bits & ~iree_vm_FeatureBits_EXT_F32;
  }
  if (bits & iree_vm_FeatureBits_EXT_F64) {
    if (length) fprintf(stdout, " | ");
    fprintf(stdout, "EXT_F64");
    length += strlen("EXT_F64");
    bits = bits & ~iree_vm_FeatureBits_EXT_F64;
  }
  if (bits) {
    if (length) fprintf(stdout, " | ");
    fprintf(stdout, "%08X", bits);
    length += 8;
  }
  return length;
}

static void iree_tooling_print_attr_def(iree_vm_AttrDef_table_t attr_def) {
  fprintf(stdout, "%s: %s", iree_vm_AttrDef_key(attr_def),
          iree_vm_AttrDef_value(attr_def));
}

static void iree_tooling_print_attr_defs(iree_vm_AttrDef_vec_t attr_defs,
                                         int indent) {
  for (size_t i = 0; i < iree_vm_AttrDef_vec_len(attr_defs); ++i) {
    iree_tooling_print_indent(indent);
    iree_tooling_print_attr_def(iree_vm_AttrDef_vec_at(attr_defs, i));
    fprintf(stdout, "\n");
  }
}

static void iree_tooling_print_type_def(iree_vm_TypeDef_table_t type_def) {
  fprintf(stdout, "%s", iree_vm_TypeDef_full_name(type_def));
}

static void iree_tooling_print_type_defs(iree_vm_TypeDef_vec_t type_defs,
                                         int indent) {
  for (size_t i = 0; i < iree_vm_TypeDef_vec_len(type_defs); ++i) {
    iree_tooling_print_indent(indent);
    fprintf(stdout, "[%3zu] ", i);
    iree_tooling_print_type_def(iree_vm_TypeDef_vec_at(type_defs, i));
    fprintf(stdout, "\n");
  }
}

static void iree_tooling_print_module_dependency_def(
    iree_vm_ModuleDependencyDef_table_t dependency_def) {
  fprintf(stdout, "%s, version >= %d, ",
          iree_vm_ModuleDependencyDef_name(dependency_def),
          iree_vm_ModuleDependencyDef_minimum_version(dependency_def));
  if (iree_vm_ModuleDependencyDef_flags(dependency_def) &
      iree_vm_ModuleDependencyFlagBits_OPTIONAL) {
    fprintf(stdout, "optional");
  } else {
    fprintf(stdout, "required");
  }
}

static void iree_tooling_print_module_dependency_defs(
    iree_vm_ModuleDependencyDef_vec_t dependency_defs, int indent) {
  for (size_t i = 0; i < iree_vm_ModuleDependencyDef_vec_len(dependency_defs);
       ++i) {
    iree_tooling_print_indent(indent);
    iree_tooling_print_module_dependency_def(
        iree_vm_ModuleDependencyDef_vec_at(dependency_defs, i));
    fprintf(stdout, "\n");
  }
}

static iree_status_t iree_tooling_print_cconv_fragment(
    iree_string_view_t cconv_fragment) {
  for (iree_host_size_t i = 0; i < cconv_fragment.size; ++i) {
    if (i > 0) fprintf(stdout, ", ");
    switch (cconv_fragment.data[i]) {
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32:
        fprintf(stdout, "i32");
        break;
      case IREE_VM_CCONV_TYPE_F32:
        fprintf(stdout, "f32");
        break;
      case IREE_VM_CCONV_TYPE_I64:
        fprintf(stdout, "i64");
        break;
      case IREE_VM_CCONV_TYPE_F64:
        fprintf(stdout, "f64");
        break;
      case IREE_VM_CCONV_TYPE_REF:
        fprintf(stdout, "!vm.ref<?>");
        break;
      case IREE_VM_CCONV_TYPE_SPAN_START: {
        iree_host_size_t end_pos = iree_string_view_find_char(
            cconv_fragment, IREE_VM_CCONV_TYPE_SPAN_END, i + 1);
        if (end_pos == IREE_STRING_VIEW_NPOS) {
          return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "unterminated cconv span");
        }
        fprintf(stdout, "tuple<");
        IREE_RETURN_IF_ERROR(iree_tooling_print_cconv_fragment(
            iree_string_view_substr(cconv_fragment, i + 1, end_pos - i - 1)));
        fprintf(stdout, ">...");
        i = end_pos;  // +1 from the for-loop
      } break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported cconv type '%c'",
                                cconv_fragment.data[i]);
    }
  }
  return iree_ok_status();
}

static void iree_tooling_print_function_cconv(flatbuffers_string_t cconv) {
  iree_vm_function_signature_t signature = {
      .calling_convention = iree_make_cstring_view(cconv),
  };
  iree_string_view_t arguments, results;
  IREE_CHECK_OK(iree_vm_function_call_get_cconv_fragments(
      &signature, &arguments, &results));
  fprintf(stdout, "(");
  IREE_CHECK_OK(iree_tooling_print_cconv_fragment(arguments));
  fprintf(stdout, ") -> (");
  IREE_CHECK_OK(iree_tooling_print_cconv_fragment(results));
  fprintf(stdout, ")");
}

static void iree_tooling_print_imported_function_def(
    iree_vm_ImportFunctionDef_table_t function_def, int indent) {
  if (iree_vm_ImportFunctionDef_flags(function_def) &
      iree_vm_ImportFlagBits_OPTIONAL) {
    fprintf(stdout, "?");
  }
  fprintf(stdout, "%s", iree_vm_ImportFunctionDef_full_name(function_def));
  iree_vm_FunctionSignatureDef_table_t signature_def =
      iree_vm_ImportFunctionDef_signature(function_def);
  if (iree_vm_FunctionSignatureDef_calling_convention_is_present(
          signature_def)) {
    iree_tooling_print_function_cconv(
        iree_vm_FunctionSignatureDef_calling_convention(signature_def));
  }
  fprintf(stdout, "\n");
  if (iree_vm_FunctionSignatureDef_attrs_is_present(signature_def)) {
    iree_tooling_print_attr_defs(
        iree_vm_FunctionSignatureDef_attrs(signature_def), indent);
  }
}

static void iree_tooling_print_imported_function_defs(
    iree_vm_ImportFunctionDef_vec_t function_defs, int indent) {
  for (size_t i = 0; i < iree_vm_ImportFunctionDef_vec_len(function_defs);
       ++i) {
    iree_tooling_print_indent(indent);
    fprintf(stdout, "[%3zu] ", i);
    iree_tooling_print_imported_function_def(
        iree_vm_ImportFunctionDef_vec_at(function_defs, i), indent + 6);
  }
}

static void iree_tooling_print_exported_function_def(
    iree_vm_ExportFunctionDef_table_t function_def,
    iree_vm_FunctionSignatureDef_vec_t function_signature_defs) {
  fprintf(stdout, "%s", iree_vm_ExportFunctionDef_local_name(function_def));
  iree_vm_FunctionSignatureDef_table_t signature_def =
      iree_vm_FunctionSignatureDef_vec_at(
          function_signature_defs,
          iree_vm_ExportFunctionDef_internal_ordinal(function_def));
  if (iree_vm_FunctionSignatureDef_calling_convention_is_present(
          signature_def)) {
    iree_tooling_print_function_cconv(
        iree_vm_FunctionSignatureDef_calling_convention(signature_def));
  }
  fprintf(stdout, "\n");
  if (iree_vm_FunctionSignatureDef_attrs_is_present(signature_def)) {
    iree_tooling_print_attr_defs(
        iree_vm_FunctionSignatureDef_attrs(signature_def), 8);
  }
}

static void iree_tooling_print_exported_function_defs(
    iree_vm_ExportFunctionDef_vec_t function_defs,
    iree_vm_FunctionSignatureDef_vec_t function_signature_defs, int indent) {
  for (size_t i = 0; i < iree_vm_ExportFunctionDef_vec_len(function_defs);
       ++i) {
    iree_tooling_print_indent(indent);
    fprintf(stdout, "[%3zu] ", i);
    iree_tooling_print_exported_function_def(
        iree_vm_ExportFunctionDef_vec_at(function_defs, i),
        function_signature_defs);
  }
}

static size_t iree_tooling_calculate_external_rodata_size(
    iree_vm_RodataSegmentDef_vec_t segment_defs) {
  size_t total_size = 0;
  for (size_t i = 0; i < iree_vm_RodataSegmentDef_vec_len(segment_defs); ++i) {
    iree_vm_RodataSegmentDef_table_t segment_def =
        iree_vm_RodataSegmentDef_vec_at(segment_defs, i);
    if (!iree_vm_RodataSegmentDef_embedded_data_is_present(segment_def)) {
      total_size += iree_vm_RodataSegmentDef_external_data_length(segment_def);
    }
  }
  return total_size;
}

// Returns true if |data| is likely a string. The string may not have a NUL
// terminator and the length is returned in |out_length|. Note that string
// tables may have NUL terminators and this will only return the first string.
static bool iree_tooling_data_is_string(flatbuffers_uint8_vec_t data,
                                        size_t* out_length) {
  *out_length = 0;
  size_t length = flatbuffers_uint8_vec_scan(data, 0);
  if (length == flatbuffers_not_found) {
    length = flatbuffers_uint8_vec_len(data);
  }
  for (size_t i = 0; i < length; ++i) {
    char c = flatbuffers_uint8_vec_at(data, i);
    if (!isprint(c)) return false;
  }
  *out_length = length;
  return true;
}

static void iree_tooling_print_rodata_segment_defs(
    iree_vm_RodataSegmentDef_vec_t segment_defs, int indent) {
  for (size_t i = 0; i < iree_vm_RodataSegmentDef_vec_len(segment_defs); ++i) {
    iree_tooling_print_indent(indent);
    iree_vm_RodataSegmentDef_table_t segment_def =
        iree_vm_RodataSegmentDef_vec_at(segment_defs, i);
    fprintf(stdout, ".rodata[%3zu] ", i);
    switch (iree_vm_RodataSegmentDef_compression_type_type(segment_def)) {
      case iree_vm_CompressionTypeDef_UncompressedDataDef:
        fprintf(stdout, "uncompressed ");
        break;
      default:
        break;
    }
    if (iree_vm_RodataSegmentDef_embedded_data_is_present(segment_def)) {
      flatbuffers_uint8_vec_t data =
          iree_vm_RodataSegmentDef_embedded_data(segment_def);
      fprintf(stdout, "embedded %8zu bytes", flatbuffers_uint8_vec_len(data));
      size_t string_length = 0;
      if (iree_tooling_data_is_string(data, &string_length)) {
        static const size_t MAX_LENGTH = 32;
        fprintf(stdout, " `%.*s`", (int)iree_min(MAX_LENGTH, string_length),
                (const char*)data);
        if (string_length >= MAX_LENGTH) fprintf(stdout, "...");
      }
    } else {
      fprintf(stdout,
              "external %8" PRIu64 " bytes (offset %" PRIu64 " / %" PRIX64
              "h to %" PRIX64 "h)",
              iree_vm_RodataSegmentDef_external_data_length(segment_def),
              iree_vm_RodataSegmentDef_external_data_offset(segment_def),
              iree_vm_RodataSegmentDef_external_data_offset(segment_def),
              iree_vm_RodataSegmentDef_external_data_offset(segment_def) +
                  iree_vm_RodataSegmentDef_external_data_length(segment_def));
    }
    fprintf(stdout, "\n");
  }
}

static void iree_tooling_print_rwdata_segment_defs(
    iree_vm_RwdataSegmentDef_vec_t segment_defs, int indent) {
  for (size_t i = 0; i < iree_vm_RwdataSegmentDef_vec_len(segment_defs); ++i) {
    iree_tooling_print_indent(indent);
    iree_vm_RwdataSegmentDef_table_t segment_def =
        iree_vm_RwdataSegmentDef_vec_at(segment_defs, i);
    fprintf(stdout, ".rwdata[%3zu] %d bytes\n", i,
            iree_vm_RwdataSegmentDef_byte_size(segment_def));
  }
}

// Returns the first export name for the given internal ordinal, or empty if not
// exported.
static iree_string_view_t iree_tooling_lookup_export_name(
    iree_host_size_t internal_ordinal,
    iree_vm_ExportFunctionDef_vec_t export_defs) {
  for (size_t j = 0; j < iree_vm_ExportFunctionDef_vec_len(export_defs); ++j) {
    iree_vm_ExportFunctionDef_table_t export_def =
        iree_vm_ExportFunctionDef_vec_at(export_defs, j);
    if (iree_vm_ExportFunctionDef_internal_ordinal(export_def) ==
        internal_ordinal) {
      const char* name = iree_vm_ExportFunctionDef_local_name(export_def);
      return iree_make_string_view(name, strlen(name));
    }
  }
  return iree_string_view_empty();
}

static void iree_tooling_print_function_descriptors(
    iree_vm_FunctionDescriptor_vec_t descriptors,
    iree_vm_ExportFunctionDef_vec_t export_defs) {
  fprintf(stdout,
          "  # | Offset   |   Length | Blocks | i32 # | ref # | Requirements | "
          "Aliases\n");
  fprintf(stdout,
          "----+----------+----------+--------+-------+-------+--------------+-"
          "----------------------------------------------------\n");
  for (size_t i = 0; i < iree_vm_FunctionDescriptor_vec_len(descriptors); ++i) {
    iree_vm_FunctionDescriptor_struct_t descriptor =
        iree_vm_FunctionDescriptor_vec_at(descriptors, i);
    fprintf(stdout, "%3zu | %08X | %8d | %6d | %5d | %5d | ", i,
            iree_vm_FunctionDescriptor_bytecode_offset(descriptor),
            iree_vm_FunctionDescriptor_bytecode_length(descriptor),
            iree_vm_FunctionDescriptor_block_count(descriptor),
            iree_vm_FunctionDescriptor_i32_register_count(descriptor),
            iree_vm_FunctionDescriptor_ref_register_count(descriptor));
    size_t pos = iree_tooling_print_feature_bits(
        iree_vm_FunctionDescriptor_requirements(descriptor));
    if (pos < 12) iree_tooling_print_indent(12 - pos);
    fprintf(stdout, " | ");
    bool any_aliases = false;
    for (size_t j = 0; j < iree_vm_ExportFunctionDef_vec_len(export_defs);
         ++j) {
      iree_vm_ExportFunctionDef_table_t export_def =
          iree_vm_ExportFunctionDef_vec_at(export_defs, j);
      if (iree_vm_ExportFunctionDef_internal_ordinal(export_def) == i) {
        if (any_aliases) fprintf(stdout, ", ");
        fprintf(stdout, "%s", iree_vm_ExportFunctionDef_local_name(export_def));
        any_aliases = true;
      }
    }
    fprintf(stdout, "\n");
  }
}

static iree_status_t iree_tooling_dump_module_metadata(
    iree_const_byte_span_t flatbuffer_contents,
    iree_const_byte_span_t rodata_contents) {
  iree_vm_BytecodeModuleDef_table_t module_def =
      iree_vm_BytecodeModuleDef_as_root(flatbuffer_contents.data);

  iree_tooling_printf_section_header(
      "@%s : version %d", iree_vm_BytecodeModuleDef_name(module_def),
      iree_vm_BytecodeModuleDef_version(module_def));
  fprintf(stdout, "\n");

  if (iree_vm_BytecodeModuleDef_requirements_is_present(module_def)) {
    fprintf(stdout, "Requirements: ");
    iree_tooling_print_feature_bits(
        iree_vm_BytecodeModuleDef_requirements(module_def));
    fprintf(stdout, "\n\n");
  }

  if (iree_vm_BytecodeModuleDef_attrs_is_present(module_def)) {
    fprintf(stdout, "Attributes:\n");
    iree_tooling_print_attr_defs(iree_vm_BytecodeModuleDef_attrs(module_def),
                                 2);
    fprintf(stdout, "\n");
  }

  if (iree_vm_BytecodeModuleDef_types_is_present(module_def)) {
    fprintf(stdout, "Required Types:\n");
    iree_tooling_print_type_defs(iree_vm_BytecodeModuleDef_types(module_def),
                                 2);
    fprintf(stdout, "\n");
  }

  if (iree_vm_BytecodeModuleDef_dependencies_is_present(module_def)) {
    fprintf(stdout, "Module Dependencies:\n");
    iree_tooling_print_module_dependency_defs(
        iree_vm_BytecodeModuleDef_dependencies(module_def), 2);
    fprintf(stdout, "\n");
  }

  if (iree_vm_BytecodeModuleDef_imported_functions_is_present(module_def)) {
    fprintf(stdout, "Imported Functions:\n");
    iree_tooling_print_imported_function_defs(
        iree_vm_BytecodeModuleDef_imported_functions(module_def), 2);
    fprintf(stdout, "\n");
  }

  if (iree_vm_BytecodeModuleDef_exported_functions_is_present(module_def)) {
    fprintf(stdout, "Exported Functions:\n");
    iree_tooling_print_exported_function_defs(
        iree_vm_BytecodeModuleDef_exported_functions(module_def),
        iree_vm_BytecodeModuleDef_function_signatures(module_def), 2);
    fprintf(stdout, "\n");
  }

  iree_tooling_printf_section_header("Sections");
  fprintf(stdout, "\n");

  if (iree_vm_BytecodeModuleDef_module_state_is_present(module_def)) {
    iree_vm_ModuleStateDef_table_t module_state_def =
        iree_vm_BytecodeModuleDef_module_state(module_def);
    fprintf(stdout, "Module State:\n");
    fprintf(stdout, "  %d bytes, %d refs, ~%zu bytes total\n",
            iree_vm_ModuleStateDef_global_bytes_capacity(module_state_def),
            iree_vm_ModuleStateDef_global_ref_count(module_state_def),
            iree_vm_ModuleStateDef_global_bytes_capacity(module_state_def) +
                sizeof(iree_vm_ref_t) *
                    iree_vm_ModuleStateDef_global_ref_count(module_state_def));
    fprintf(stdout, "\n");
  }

  fprintf(stdout, "FlatBuffer: %" PRIhsz " bytes\n",
          flatbuffer_contents.data_length);
  if (iree_vm_BytecodeModuleDef_bytecode_data_is_present(module_def)) {
    flatbuffers_uint8_vec_t bytecode_data =
        iree_vm_BytecodeModuleDef_bytecode_data(module_def);
    fprintf(stdout, "  Bytecode: %zu bytes\n",
            flatbuffers_uint8_vec_len(bytecode_data));
  }
  if (iree_vm_BytecodeModuleDef_rodata_segments_is_present(module_def)) {
    iree_tooling_print_rodata_segment_defs(
        iree_vm_BytecodeModuleDef_rodata_segments(module_def), 2);
  }
  if (iree_vm_BytecodeModuleDef_rwdata_segments_is_present(module_def)) {
    iree_tooling_print_rwdata_segment_defs(
        iree_vm_BytecodeModuleDef_rwdata_segments(module_def), 2);
  }
  fprintf(stdout, "\n");
  size_t external_size = iree_tooling_calculate_external_rodata_size(
      iree_vm_BytecodeModuleDef_rodata_segments(module_def));
  if (external_size > 0) {
    fprintf(stdout, "External .rodata: ~%zu bytes\n\n", external_size);
  }

  iree_tooling_printf_section_header(
      "Bytecode : version %d", iree_vm_BytecodeModuleDef_version(module_def));
  fprintf(stdout, "\n");

  if (iree_vm_BytecodeModuleDef_function_descriptors_is_present(module_def)) {
    iree_tooling_print_function_descriptors(
        iree_vm_BytecodeModuleDef_function_descriptors(module_def),
        iree_vm_BytecodeModuleDef_exported_functions(module_def));
  }

  fprintf(stdout, "\n");

  if (iree_vm_BytecodeModuleDef_debug_database_is_present(module_def)) {
    iree_tooling_printf_section_header("Debug Information");
    iree_vm_DebugDatabaseDef_table_t database_def =
        iree_vm_BytecodeModuleDef_debug_database(module_def);
    fprintf(stdout,
            "// NOTE: debug databases are large and should be stripped in "
            "deployed artifacts.\n");
    fprintf(stdout, "\n");

    // TODO(benvanik): some kind of informative output; for now just counts to
    // indicate how much is present (would be nice to calculate full sizing).
    fprintf(stdout, "Locations: %zu\n",
            flatbuffers_generic_vec_len(
                iree_vm_DebugDatabaseDef_location_table(database_def)));
    fprintf(stdout, "\n");
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// --output=flatbuffer-binary
//===----------------------------------------------------------------------===//

static iree_status_t iree_tooling_dump_module_flatbuffer_binary(
    iree_const_byte_span_t flatbuffer_contents) {
  IREE_IO_SET_BINARY_MODE(stdout);  // ensure binary output mode
  fwrite(flatbuffer_contents.data, 1, flatbuffer_contents.data_length, stdout);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// --output=flatbuffer-json
//===----------------------------------------------------------------------===//

IREE_FLAG(bool, pretty, false,
          "Enable indentation when outputting flatbuffer JSON.");

static iree_status_t iree_tooling_dump_module_flatbuffer_json(
    iree_const_byte_span_t flatbuffer_contents) {
  flatcc_json_printer_t printer;
  flatcc_json_printer_init(&printer, NULL);
  if (FLAG_pretty) {
    flatcc_json_printer_set_flags(&printer, flatcc_json_printer_f_pretty);
  }
  flatcc_json_printer_set_skip_default(&printer, true);
  bytecode_module_def_print_json(&printer,
                                 (const char*)flatbuffer_contents.data,
                                 flatbuffer_contents.data_length);
  flatcc_json_printer_flush(&printer);
  flatcc_json_printer_clear(&printer);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// --output=disassembly
//===----------------------------------------------------------------------===//

static iree_status_t iree_tooling_dump_module_disassembly(
    iree_const_byte_span_t archive_contents, iree_string_view_t function_filter,
    iree_allocator_t host_allocator) {
  // Create a VM instance for type registration.
  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                               host_allocator, &instance));

  // Load module with placeholder types (for disassembly, we don't need real
  // type registrations).
  iree_vm_module_t* module = NULL;
  iree_status_t status = iree_vm_bytecode_module_create(
      instance, IREE_VM_BYTECODE_MODULE_FLAG_ALLOW_PLACEHOLDER_TYPES,
      archive_contents, iree_allocator_null(), host_allocator, &module);
  iree_const_byte_span_t flatbuffer_contents = iree_const_byte_span_empty();
  iree_host_size_t rodata_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_archive_parse_header(
        archive_contents, &flatbuffer_contents, &rodata_offset);
  }
  if (iree_status_is_ok(status)) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(host_allocator, &builder);

    // Extract export names from the flatbuffer module definition.
    iree_vm_BytecodeModuleDef_table_t module_def =
        iree_vm_BytecodeModuleDef_as_root(flatbuffer_contents.data);
    iree_vm_ExportFunctionDef_vec_t export_defs =
        iree_vm_BytecodeModuleDef_exported_functions(module_def);

    // Iterate over internal functions and build the disassembly output.
    iree_vm_module_signature_t signature = iree_vm_module_signature(module);
    for (iree_host_size_t i = 0; i < signature.internal_function_count; ++i) {
      iree_vm_function_t function;
      status = iree_vm_module_lookup_function_by_ordinal(
          module, IREE_VM_FUNCTION_LINKAGE_INTERNAL, i, &function);
      if (!iree_status_is_ok(status)) break;

      // Get function name from exports if available, otherwise use internal
      // name.
      iree_string_view_t export_name =
          iree_tooling_lookup_export_name(i, export_defs);
      iree_string_view_t function_name = iree_string_view_is_empty(export_name)
                                             ? iree_vm_function_name(&function)
                                             : export_name;

      // Apply filter (ordinal or name) if provided.
      if (!iree_string_view_is_empty(function_filter)) {
        uint32_t filter_ordinal = -1;
        if (iree_string_view_atoi_uint32(function_filter, &filter_ordinal)) {
          if (i != filter_ordinal) continue;
        } else if (!iree_string_view_equal(function_name, function_filter)) {
          continue;
        }
      }

      // Print function header.
      status = iree_string_builder_append_string(
          &builder, IREE_SV("//"
                            "===-----------------------------------------------"
                            "-----------------------===//\n"));
      if (!iree_status_is_ok(status)) break;
      iree_string_view_t module_name = iree_vm_module_name(module);
      status = iree_string_builder_append_format(
          &builder, "// Disassembly: %.*s.%.*s (#%" PRIhsz ")\n",
          (int)module_name.size, module_name.data, (int)function_name.size,
          function_name.data, i);
      if (!iree_status_is_ok(status)) break;
      status = iree_string_builder_append_string(
          &builder, IREE_SV("//"
                            "===-----------------------------------------------"
                            "-----------------------===//\n"));
      if (!iree_status_is_ok(status)) break;

      // Disassemble the function.
      status =
          iree_vm_bytecode_module_disassemble_function(module, i, &builder);
      if (!iree_status_is_ok(status)) break;

      // Flush output to stdout.
      fprintf(stdout, "%.*s\n", (int)iree_string_builder_size(&builder),
              iree_string_builder_buffer(&builder));
      iree_string_builder_reset(&builder);
    }

    iree_string_builder_deinitialize(&builder);
  }

  iree_vm_module_release(module);
  iree_vm_instance_release(instance);
  return status;
}

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

IREE_FLAG(string, output, "all",
          "Output mode:\n"
          "  'all': metadata and disassembly.\n"
          "  'metadata': module metadata and size breakdown.\n"
          "  'disassembly': bytecode disassembly of all functions.\n"
          "  'flatbuffer-binary': module flatbuffer in binary format.\n"
          "  'flatbuffer-json': module flatbuffer in JSON format.\n");

IREE_FLAG(string, function, "",
          "Filter disassembly to a specific function name (exact match).\n"
          "Only used with --output=disassembly.");

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  // Parse command line flags.
  iree_flags_set_usage(
      "iree-dump-module",
      "Dumps IREE VM module details to stdout.\n"
      "$ iree-dump-module [--output=...] module.vmfb\n"
      "\n"
      "Example dumping all information (metadata + disassembly):\n"
      "  iree-dump-module module.vmfb\n"
      "  (or)\n"
      "  iree-dump-module --output=all module.vmfb\n"
      "\n"
      "Example dumping metadata (types, imports, exports, custom attributes):\n"
      "  iree-dump-module --output=metadata module.vmfb\n"
      "\n"
      "Example dumping VM ISA disassembly for all functions:\n"
      "  iree-dump-module --output=disassembly module.vmfb\n"
      "\n"
      "Example dumping VM ISA disassembly for a single function by name:\n"
      "  iree-dump-module --output=disassembly --function=fn2 module.vmfb\n"
      "And internal function ordinal:\n"
      "  iree-dump-module --output=disassembly --function=24 module.vmfb\n"
      "\n"
      "Dump flatbuffer binary or JSON:\n"
      "  iree-dump-module --output=flatbuffer-binary module.vmfb\n"
      "  iree-dump-module --output=flatbuffer-json --pretty module.vmfb\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  if (argc < 2) {
    fprintf(stderr, "Syntax: iree-dump-module [--output=...] module.vmfb\n");
    IREE_TRACE_APP_EXIT(EXIT_FAILURE);
    return EXIT_FAILURE;
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      iree_make_cstring_view(argv[1]), IREE_IO_FILE_ACCESS_READ, host_allocator,
      &file_contents);

  iree_const_byte_span_t flatbuffer_contents = iree_const_byte_span_empty();
  iree_const_byte_span_t rodata_contents = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    iree_host_size_t rodata_offset = 0;
    status = iree_vm_bytecode_archive_parse_header(
        file_contents->const_buffer, &flatbuffer_contents, &rodata_offset);
    rodata_contents = iree_make_const_byte_span(
        file_contents->const_buffer.data + rodata_offset,
        file_contents->const_buffer.data_length - rodata_offset);
  }
  if (iree_status_is_ok(status)) {
    int verify_ret = iree_vm_BytecodeModuleDef_verify_as_root(
        flatbuffer_contents.data, flatbuffer_contents.data_length);
    if (verify_ret != flatcc_verify_ok) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "FlatBuffer verification failed: %s",
                                flatcc_verify_error_string(verify_ret));
    }
  }
  if (iree_status_is_ok(status)) {
    if (strcmp(FLAG_output, "all") == 0) {
      status = iree_tooling_dump_module_metadata(flatbuffer_contents,
                                                 rodata_contents);
      if (iree_status_is_ok(status)) {
        fprintf(stdout, "\n");
        status = iree_tooling_dump_module_disassembly(
            file_contents->const_buffer, iree_make_cstring_view(FLAG_function),
            host_allocator);
      }
    } else if (strcmp(FLAG_output, "metadata") == 0) {
      status = iree_tooling_dump_module_metadata(flatbuffer_contents,
                                                 rodata_contents);
    } else if (strcmp(FLAG_output, "disassembly") == 0) {
      status = iree_tooling_dump_module_disassembly(
          file_contents->const_buffer, iree_make_cstring_view(FLAG_function),
          host_allocator);
    } else if (strcmp(FLAG_output, "flatbuffer-binary") == 0) {
      status = iree_tooling_dump_module_flatbuffer_binary(flatbuffer_contents);
    } else if (strcmp(FLAG_output, "flatbuffer-json") == 0) {
      status = iree_tooling_dump_module_flatbuffer_json(flatbuffer_contents);
    } else {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unrecognized --output= flag value '%s'",
                                FLAG_output);
    }
  }

  iree_io_file_contents_free(file_contents);

  fflush(stdout);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
