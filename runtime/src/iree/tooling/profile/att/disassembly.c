// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/att/disassembly.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/tooling/profile/att/util.h"

typedef struct iree_profile_att_elf64_header_t {
  // ELF identification bytes.
  uint8_t ident[16];
  // Object file type.
  uint16_t type;
  // Target machine.
  uint16_t machine;
  // ELF object version.
  uint32_t version;
  // Entry point virtual address.
  uint64_t entry;
  // Program header table file offset.
  uint64_t program_header_offset;
  // Section header table file offset.
  uint64_t section_header_offset;
  // Processor-specific flags.
  uint32_t flags;
  // ELF header byte size.
  uint16_t header_size;
  // Program header entry byte size.
  uint16_t program_header_entry_size;
  // Program header entry count.
  uint16_t program_header_count;
  // Section header entry byte size.
  uint16_t section_header_entry_size;
  // Section header entry count.
  uint16_t section_header_count;
  // Section name string table index.
  uint16_t section_name_string_table_index;
} iree_profile_att_elf64_header_t;

typedef struct iree_profile_att_elf64_program_header_t {
  // Program header segment type.
  uint32_t type;
  // Program header flags.
  uint32_t flags;
  // Segment file offset.
  uint64_t offset;
  // Segment virtual address.
  uint64_t virtual_address;
  // Segment physical address.
  uint64_t physical_address;
  // Segment byte length in the file.
  uint64_t file_size;
  // Segment byte length in memory.
  uint64_t memory_size;
  // Segment alignment.
  uint64_t alignment;
} iree_profile_att_elf64_program_header_t;

enum {
  IREE_PROFILE_ATT_ELF_MAGIC0 = 0x7F,
  IREE_PROFILE_ATT_ELF_MAGIC1 = 'E',
  IREE_PROFILE_ATT_ELF_MAGIC2 = 'L',
  IREE_PROFILE_ATT_ELF_MAGIC3 = 'F',
  IREE_PROFILE_ATT_ELF_CLASS_64 = 2,
  IREE_PROFILE_ATT_ELF_DATA_LITTLE = 1,
  IREE_PROFILE_ATT_ELF_PROGRAM_HEADER_LOAD = 1,
};

typedef struct iree_profile_att_code_object_decoder_t {
  // Producer-local code-object marker identifier.
  uint64_t code_object_id;
  // Borrowed exact HSACO bytes from the mapped profile bundle.
  iree_const_byte_span_t data;
  // AMD COMGR data object holding |data|.
  iree_profile_att_comgr_data_t comgr_data;
  // AMD COMGR disassembly context for the code object's ISA.
  iree_profile_att_comgr_disassembly_info_t disassembly_info;
} iree_profile_att_code_object_decoder_t;

struct iree_profile_att_disassembly_context_t {
  // Host allocator used for dynamic code-object decoder arrays.
  iree_allocator_t host_allocator;
  // AMD COMGR library table used for disassembly.
  const iree_profile_att_comgr_library_t* comgr;
  // Dynamic array of disassembly contexts, one per loaded code object.
  iree_profile_att_code_object_decoder_t* code_object_decoders;
  // Number of valid entries in |code_object_decoders|.
  iree_host_size_t code_object_decoder_count;
  // Capacity of |code_object_decoders| in entries.
  iree_host_size_t code_object_decoder_capacity;
};

typedef struct iree_profile_att_comgr_disassemble_context_t {
  // Borrowed code-object bytes being disassembled.
  iree_const_byte_span_t image;
  // Last instruction text printed by AMD COMGR.
  char instruction[1024];
} iree_profile_att_comgr_disassemble_context_t;

static bool iree_profile_att_elf_virtual_address_to_file_offset(
    iree_const_byte_span_t image, uint64_t virtual_address,
    uint64_t* out_file_offset) {
  *out_file_offset = 0;
  if (image.data_length < sizeof(iree_profile_att_elf64_header_t)) {
    return false;
  }

  iree_profile_att_elf64_header_t header;
  memcpy(&header, image.data, sizeof(header));
  if (header.ident[0] != IREE_PROFILE_ATT_ELF_MAGIC0 ||
      header.ident[1] != IREE_PROFILE_ATT_ELF_MAGIC1 ||
      header.ident[2] != IREE_PROFILE_ATT_ELF_MAGIC2 ||
      header.ident[3] != IREE_PROFILE_ATT_ELF_MAGIC3 ||
      header.ident[4] != IREE_PROFILE_ATT_ELF_CLASS_64 ||
      header.ident[5] != IREE_PROFILE_ATT_ELF_DATA_LITTLE) {
    return false;
  }
  if (header.program_header_entry_size <
      sizeof(iree_profile_att_elf64_program_header_t)) {
    return false;
  }
  if (header.program_header_offset > image.data_length) {
    return false;
  }

  const uint64_t table_length =
      (uint64_t)header.program_header_entry_size * header.program_header_count;
  if (table_length > image.data_length - header.program_header_offset) {
    return false;
  }

  for (uint16_t i = 0; i < header.program_header_count; ++i) {
    const uint64_t program_header_offset =
        header.program_header_offset +
        (uint64_t)i * header.program_header_entry_size;
    iree_profile_att_elf64_program_header_t program_header;
    memcpy(&program_header, image.data + program_header_offset,
           sizeof(program_header));
    if (program_header.type != IREE_PROFILE_ATT_ELF_PROGRAM_HEADER_LOAD) {
      continue;
    }
    if (virtual_address < program_header.virtual_address) continue;
    const uint64_t segment_offset =
        virtual_address - program_header.virtual_address;
    if (segment_offset >= program_header.file_size) continue;
    if (program_header.offset > image.data_length ||
        segment_offset > image.data_length - program_header.offset) {
      return false;
    }
    *out_file_offset = program_header.offset + segment_offset;
    return true;
  }
  return false;
}

static uint64_t iree_profile_att_comgr_read_memory(uint64_t from, char* to,
                                                   uint64_t size,
                                                   void* user_data) {
  iree_profile_att_comgr_disassemble_context_t* context =
      (iree_profile_att_comgr_disassemble_context_t*)user_data;
  const uintptr_t begin = (uintptr_t)context->image.data;
  const uintptr_t end = begin + context->image.data_length;
  if (from < begin || from >= end) return 0;
  const uint64_t available = end - from;
  const uint64_t read_length = iree_min(size, available);
  memcpy(to, (const void*)(uintptr_t)from, (size_t)read_length);
  return read_length;
}

static void iree_profile_att_comgr_print_instruction(const char* instruction,
                                                     void* user_data) {
  iree_profile_att_comgr_disassemble_context_t* context =
      (iree_profile_att_comgr_disassemble_context_t*)user_data;
  iree_string_view_to_cstring(
      iree_profile_att_cstring_view_or_empty(instruction), context->instruction,
      sizeof(context->instruction));
}

static void iree_profile_att_comgr_print_address_annotation(uint64_t address,
                                                            void* user_data) {
  (void)address;
  (void)user_data;
}

static iree_status_t iree_profile_att_code_object_decoder_initialize(
    const iree_profile_att_comgr_library_t* comgr,
    const iree_profile_att_code_object_t* code_object,
    iree_profile_att_code_object_decoder_t* out_decoder) {
  memset(out_decoder, 0, sizeof(*out_decoder));
  out_decoder->code_object_id = code_object->code_object_id;
  out_decoder->data = code_object->data;

  iree_status_t status = iree_profile_att_make_comgr_status(
      comgr,
      comgr->create_data(IREE_PROFILE_ATT_COMGR_DATA_KIND_EXECUTABLE,
                         &out_decoder->comgr_data),
      "amd_comgr_create_data");
  if (iree_status_is_ok(status)) {
    status = iree_profile_att_make_comgr_status(
        comgr,
        comgr->set_data(out_decoder->comgr_data, out_decoder->data.data_length,
                        (const char*)out_decoder->data.data),
        "amd_comgr_set_data");
  }

  char isa_name[256];
  memset(isa_name, 0, sizeof(isa_name));
  if (iree_status_is_ok(status)) {
    size_t isa_name_length = sizeof(isa_name);
    status = iree_profile_att_make_comgr_status(
        comgr,
        comgr->get_data_isa_name(out_decoder->comgr_data, &isa_name_length,
                                 isa_name),
        "amd_comgr_get_data_isa_name");
  }
  if (iree_status_is_ok(status)) {
    status = iree_profile_att_make_comgr_status(
        comgr,
        comgr->create_disassembly_info(
            isa_name, iree_profile_att_comgr_read_memory,
            iree_profile_att_comgr_print_instruction,
            iree_profile_att_comgr_print_address_annotation,
            &out_decoder->disassembly_info),
        "amd_comgr_create_disassembly_info");
  }

  if (!iree_status_is_ok(status)) {
    if (out_decoder->disassembly_info.handle) {
      comgr->destroy_disassembly_info(out_decoder->disassembly_info);
    }
    if (out_decoder->comgr_data.handle) {
      comgr->release_data(out_decoder->comgr_data);
    }
    memset(out_decoder, 0, sizeof(*out_decoder));
  }
  return status;
}

static void iree_profile_att_code_object_decoder_deinitialize(
    const iree_profile_att_comgr_library_t* comgr,
    iree_profile_att_code_object_decoder_t* decoder) {
  if (decoder->disassembly_info.handle) {
    comgr->destroy_disassembly_info(decoder->disassembly_info);
  }
  if (decoder->comgr_data.handle) {
    comgr->release_data(decoder->comgr_data);
  }
  memset(decoder, 0, sizeof(*decoder));
}

static iree_profile_att_code_object_decoder_t*
iree_profile_att_disassembly_context_find_code_object(
    const iree_profile_att_disassembly_context_t* context,
    uint64_t code_object_id) {
  for (iree_host_size_t i = 0; i < context->code_object_decoder_count; ++i) {
    iree_profile_att_code_object_decoder_t* decoder =
        &context->code_object_decoders[i];
    if (decoder->code_object_id == code_object_id) return decoder;
  }
  return NULL;
}

iree_status_t iree_profile_att_disassembly_context_allocate(
    iree_allocator_t host_allocator,
    const iree_profile_att_comgr_library_t* comgr,
    iree_profile_att_disassembly_context_t** out_context) {
  *out_context = NULL;
  iree_profile_att_disassembly_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*context),
                                             (void**)&context));
  memset(context, 0, sizeof(*context));
  context->host_allocator = host_allocator;
  context->comgr = comgr;
  *out_context = context;
  return iree_ok_status();
}

void iree_profile_att_disassembly_context_free(
    iree_profile_att_disassembly_context_t* context) {
  if (!context) return;
  for (iree_host_size_t i = 0; i < context->code_object_decoder_count; ++i) {
    iree_profile_att_code_object_decoder_deinitialize(
        context->comgr, &context->code_object_decoders[i]);
  }
  iree_allocator_t host_allocator = context->host_allocator;
  iree_allocator_free(host_allocator, context->code_object_decoders);
  iree_allocator_free(host_allocator, context);
}

iree_status_t iree_profile_att_disassembly_context_ensure_code_object_loaded(
    iree_profile_att_disassembly_context_t* context,
    const iree_profile_att_code_object_t* code_object) {
  if (iree_profile_att_disassembly_context_find_code_object(
          context, code_object->code_object_id)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_profile_att_grow_array(
      context->host_allocator, context->code_object_decoder_count + 1,
      sizeof(context->code_object_decoders[0]),
      &context->code_object_decoder_capacity,
      (void**)&context->code_object_decoders));
  iree_profile_att_code_object_decoder_t* decoder =
      &context->code_object_decoders[context->code_object_decoder_count];
  IREE_RETURN_IF_ERROR(iree_profile_att_code_object_decoder_initialize(
      context->comgr, code_object, decoder));
  ++context->code_object_decoder_count;
  return iree_ok_status();
}

iree_status_t iree_profile_att_disassembly_context_disassemble_instruction(
    iree_profile_att_disassembly_context_t* context,
    const iree_profile_att_pc_key_t* pc, char* instruction_buffer,
    iree_host_size_t instruction_buffer_capacity,
    iree_string_view_t* out_instruction) {
  instruction_buffer[0] = 0;
  *out_instruction = iree_string_view_empty();

  iree_profile_att_code_object_decoder_t* decoder =
      iree_profile_att_disassembly_context_find_code_object(context,
                                                            pc->code_object_id);
  if (!decoder) return iree_ok_status();

  uint64_t file_offset = 0;
  if (!iree_profile_att_elf_virtual_address_to_file_offset(
          decoder->data, pc->address, &file_offset)) {
    return iree_ok_status();
  }
  if (file_offset >= decoder->data.data_length) return iree_ok_status();

  iree_profile_att_comgr_disassemble_context_t callback_context = {
      .image = decoder->data,
  };
  const uint64_t address =
      (uint64_t)(uintptr_t)(decoder->data.data + file_offset);
  uint64_t instruction_size = 0;
  iree_status_t status = iree_profile_att_make_comgr_status(
      context->comgr,
      context->comgr->disassemble_instruction(decoder->disassembly_info,
                                              address, &callback_context,
                                              &instruction_size),
      "amd_comgr_disassemble_instruction");
  if (iree_status_is_ok(status)) {
    iree_string_view_to_cstring(
        iree_make_cstring_view(callback_context.instruction),
        instruction_buffer, instruction_buffer_capacity);
    *out_instruction = iree_make_cstring_view(instruction_buffer);
  }
  return status;
}
