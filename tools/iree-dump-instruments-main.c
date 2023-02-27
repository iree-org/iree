// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/schemas/instruments/dispatch.h"

// NOTE: include order matters:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/instruments/dispatch_def_reader.h"

typedef struct {
  iree_instruments_DispatchFunctionDef_vec_t functions_def;
  iree_instruments_DispatchSiteDef_vec_t dispatch_sites_def;
} iree_dispatch_metadata_t;

static iree_status_t iree_tooling_dump_dispatch_metadata(
    const uint8_t* flatbuffer_ptr, iree_host_size_t flatbuffer_size,
    iree_dispatch_metadata_t* out_metadata, FILE* stream) {
  memset(out_metadata, 0, sizeof(*out_metadata));

  iree_instruments_DispatchInstrumentDef_table_t instr_def =
      iree_instruments_DispatchInstrumentDef_as_root(flatbuffer_ptr);

  iree_instruments_DispatchFunctionDef_vec_t functions_def =
      iree_instruments_DispatchInstrumentDef_functions(instr_def);
  out_metadata->functions_def = functions_def;
  for (iree_host_size_t i = 0;
       i < iree_instruments_DispatchFunctionDef_vec_len(functions_def); ++i) {
    fprintf(stream, "\n");
    iree_instruments_DispatchFunctionDef_table_t function_def =
        iree_instruments_DispatchFunctionDef_vec_at(functions_def, i);
    flatbuffers_string_t name =
        iree_instruments_DispatchFunctionDef_name(function_def);
    fprintf(stream,
            "//"
            "===---------------------------------------------------------------"
            "-------===//\n");
    fprintf(stream, "// export[%" PRIhsz "]: %s\n", i, name);
    fprintf(stream,
            "//"
            "===---------------------------------------------------------------"
            "-------===//\n");
    flatbuffers_string_t target =
        iree_instruments_DispatchFunctionDef_target(function_def);
    if (target) fprintf(stream, "//  target: %s\n", target);
    flatbuffers_string_t layout =
        iree_instruments_DispatchFunctionDef_layout(function_def);
    if (layout) fprintf(stream, "//  layout: %s\n", layout);
    flatbuffers_string_t source =
        iree_instruments_DispatchFunctionDef_source(function_def);
    if (source) fprintf(stream, "%s\n", source);
    fprintf(stream, "\n");
  }

  fprintf(stream,
          "//"
          "===---------------------------------------------------------------"
          "-------===//\n");
  iree_instruments_DispatchSiteDef_vec_t dispatch_sites_def =
      iree_instruments_DispatchInstrumentDef_sites(instr_def);
  out_metadata->dispatch_sites_def = dispatch_sites_def;
  for (iree_host_size_t i = 0;
       i < iree_instruments_DispatchSiteDef_vec_len(dispatch_sites_def); ++i) {
    iree_instruments_DispatchSiteDef_table_t dispatch_site_def =
        iree_instruments_DispatchSiteDef_vec_at(dispatch_sites_def, i);
    iree_instruments_DispatchFunctionDef_table_t function_def =
        iree_instruments_DispatchFunctionDef_vec_at(
            functions_def,
            iree_instruments_DispatchSiteDef_function(dispatch_site_def));
    flatbuffers_string_t name =
        iree_instruments_DispatchFunctionDef_name(function_def);
    fprintf(stream, "// dispatch site %" PRIhsz ": %s\n", i, name);
  }
  fprintf(stream,
          "//"
          "===---------------------------------------------------------------"
          "-------===//\n\n");

  return iree_ok_status();
}

static void iree_tooling_dump_print_value(
    iree_instrument_dispatch_value_type_t type, uint64_t raw_value,
    FILE* stream) {
  union {
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f32;
    double f64;
    uint8_t value_storage[sizeof(uint64_t)];
  } value = {.i64 = raw_value};
  switch (type) {
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_8:
      fprintf(stream, "%" PRId8, value.i8);
      break;
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_8:
      fprintf(stream, "%" PRIu8, value.i8);
      break;
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_16:
      fprintf(stream, "%" PRId16, value.i16);
      break;
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_16:
      fprintf(stream, "%" PRIu16, value.i16);
      break;
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_32:
      fprintf(stream, "%" PRId32, value.i32);
      break;
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_32:
      fprintf(stream, "%" PRIu32, value.i32);
      break;
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_64:
      fprintf(stream, "%" PRId64, value.i64);
      break;
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_64:
      fprintf(stream, "%" PRIu64, value.i64);
      break;
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_POINTER:
      fprintf(stream, "%16" PRIX64, value.i64);
      break;
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_16:
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_BFLOAT_16:
      fprintf(stream, "%4" PRIX16, value.i16);
      break;
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_32:
      fprintf(stream, "%e %f", value.f32, value.f32);
      break;
    case IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_64:
      fprintf(stream, "%e %f", value.f64, value.f64);
      break;
    default:
      fprintf(stream, "<<unknown type: %02X>>", (uint32_t)type);
      break;
  }
}

static iree_status_t iree_tooling_dump_dispatch_ringbuffer(
    const uint8_t* data_ptr, iree_host_size_t data_size,
    const iree_dispatch_metadata_t* metadata, FILE* stream) {
  const uint64_t ring_size = data_size - IREE_INSTRUMENT_DISPATCH_PADDING;
  const uint8_t* ring_data = data_ptr;
  const uint64_t ring_head = *(const uint64_t*)(ring_data + data_size - 8);
  const uint64_t ring_range = iree_min(ring_head, ring_size);

  for (iree_host_size_t i = 0; i < ring_range;) {
    const iree_instrument_dispatch_header_t* header =
        (const iree_instrument_dispatch_header_t*)(ring_data + i);
    switch (header->tag) {
      case IREE_INSTRUMENT_DISPATCH_TYPE_WORKGROUP: {
        const iree_instrument_dispatch_workgroup_t* workgroup =
            (const iree_instrument_dispatch_workgroup_t*)header;
        iree_instruments_DispatchSiteDef_table_t dispatch_site_def =
            iree_instruments_DispatchSiteDef_vec_at(
                metadata->dispatch_sites_def, workgroup->dispatch_id);
        iree_instruments_DispatchFunctionDef_table_t function_def =
            iree_instruments_DispatchFunctionDef_vec_at(
                metadata->functions_def,
                iree_instruments_DispatchSiteDef_function(dispatch_site_def));
        flatbuffers_string_t name_def =
            iree_instruments_DispatchFunctionDef_name(function_def);
        fprintf(stream,
                "%016" PRIX64
                " | WORKGROUP dispatch(%u %s %ux%ux%u) %u,%u,%u pid:%u\n",
                (uint64_t)i, workgroup->dispatch_id, name_def,
                workgroup->workgroup_count_x, workgroup->workgroup_count_y,
                workgroup->workgroup_count_z, workgroup->workgroup_id_x,
                workgroup->workgroup_id_y, workgroup->workgroup_id_z,
                workgroup->processor_id);
        i += sizeof(*workgroup);
        break;
      }
      case IREE_INSTRUMENT_DISPATCH_TYPE_PRINT: {
        const iree_instrument_dispatch_print_t* print =
            (const iree_instrument_dispatch_print_t*)header;
        fprintf(stream, "%016" PRIX64 " | PRINT %.*s\n",
                print->workgroup_offset, (int)print->length, print->data);
        i += iree_host_align(sizeof(*print) + print->length, 16);
        break;
      }
      case IREE_INSTRUMENT_DISPATCH_TYPE_VALUE: {
        const iree_instrument_dispatch_value_t* value =
            (const iree_instrument_dispatch_value_t*)header;
        fprintf(stream,
                "%016" PRIX64 " | VALUE %04u = ", value->workgroup_offset,
                (uint32_t)value->ordinal);
        iree_tooling_dump_print_value(value->type, value->bits, stream);
        fputc('\n', stream);
        i += sizeof(*value);
        break;
      }
      case IREE_INSTRUMENT_DISPATCH_TYPE_MEMORY_LOAD: {
        const iree_instrument_dispatch_memory_op_t* op =
            (const iree_instrument_dispatch_memory_op_t*)header;
        fprintf(stream, "%016" PRIX64 " | LOAD  %016" PRIX64 " %u\n",
                op->workgroup_offset, op->address, (int)op->length);
        i += sizeof(*op);
        break;
      }
      case IREE_INSTRUMENT_DISPATCH_TYPE_MEMORY_STORE: {
        const iree_instrument_dispatch_memory_op_t* op =
            (const iree_instrument_dispatch_memory_op_t*)header;
        fprintf(stream, "%016" PRIX64 " | STORE %016" PRIX64 " %u\n",
                op->workgroup_offset, op->address, (int)op->length);
        i += sizeof(*op);
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unimplemented dispatch instr type: %u",
                                (uint32_t)header->tag);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_tooling_dump_instrument_file(
    iree_const_byte_span_t file_contents, FILE* stream) {
  const uint8_t* file_ptr = file_contents.data;
  iree_host_size_t file_size = file_contents.data_length;

  iree_dispatch_metadata_t dispatch_metadata = {0};
  for (iree_host_size_t file_offset = 0; file_offset < file_size;) {
    const iree_idbts_chunk_header_t* header =
        (const iree_idbts_chunk_header_t*)(file_ptr + file_offset);
    const uint8_t* payload = file_ptr + file_offset + sizeof(*header);
    switch (header->type) {
      case IREE_IDBTS_CHUNK_TYPE_DISPATCH_METADATA: {
        IREE_RETURN_IF_ERROR(iree_tooling_dump_dispatch_metadata(
            payload, header->content_length, &dispatch_metadata, stream));
        break;
      }
      case IREE_IDBTS_CHUNK_TYPE_DISPATCH_RINGBUFFER: {
        IREE_RETURN_IF_ERROR(iree_tooling_dump_dispatch_ringbuffer(
            payload, header->content_length, &dispatch_metadata, stream));
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unimplemented chunk type: %u",
                                (uint32_t)header->type);
    }
    file_offset +=
        sizeof(*header) + iree_host_align(header->content_length, 16);
  }

  return iree_ok_status();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Syntax: iree-dump-instruments instruments.bin > instruments.txt\n"
            "Example usage:\n"
            "  $ iree-compile \\n"
            "        --iree-hal-target-backends=llvm-cpu \\n"
            "        --iree-hal-instrument-dispatches=16mib \\n"
            "        --iree-llvmcpu-instrument-memory-accesses=false \\n"
            "        runtime/src/iree/runtime/testdata/simple_mul.mlir \\n"
            "        -o=simple_mul_instr.vmfb\n"
            "  $ iree-run-module \\n"
            "        --device=local-sync \\n"
            "        --module=simple_mul_instr.vmfb \\n"
            "        --function=simple_mul \\n"
            "        --input=4xf32=2 \\n"
            "        --input=4xf32=4 \\n"
            "        --instrument_file=instrument.bin\n"
            "  $ iree-dump-instruments instrument.bin\n"
            "\n");
    return 1;
  }

  iree_file_contents_t* file_contents = NULL;
  iree_status_t status =
      iree_file_read_contents(argv[1], iree_allocator_system(), &file_contents);
  if (iree_status_is_ok(status)) {
    status =
        iree_tooling_dump_instrument_file(file_contents->const_buffer, stdout);
  }
  iree_file_contents_free(file_contents);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
