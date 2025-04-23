#include <fstream>
#include <iostream>

#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/exsleratev2_executable_def_builder.h"
#include "iree/schemas/exsleratev2_executable_def_reader.h"
#include "iree/schemas/exsleratev2_executable_def_verifier.h"

void deserializeFromSLFb(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    printf("Error opening file for reading\n");
    return;
  }

  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  void *buffer = malloc(size);
  if (fread(buffer, 1, size, fp) != size) {
    printf("Error reading file\n");
    fclose(fp);
    free(buffer);
    return;
  }
  fclose(fp);

  iree_exsleratev2_hal_exsleratev2_ExecutableDef_table_t executable =
      iree_exsleratev2_hal_exsleratev2_ExecutableDef_as_root(buffer);

  if (!executable) {
    printf("Invalid FlatBuffer format\n");
    free(buffer);
    return;
  }

  flatbuffers_string_vec_t entry_points =
      iree_exsleratev2_hal_exsleratev2_ExecutableDef_entry_points(executable);
  if (entry_points) {
    size_t count = flatbuffers_string_vec_len(entry_points);
    printf("Found %zu entry points:\n", count);
    for (size_t i = 0; i < count; i++) {
      const char *name = flatbuffers_string_vec_at(entry_points, i);
      printf("  %zu: %s\n", i, name ? name : "(null)");
    }
  }

  iree_exsleratev2_hal_exsleratev2_LayerDef_vec_t layers =
      iree_exsleratev2_hal_exsleratev2_ExecutableDef_layers(executable);
  if (!layers) {
    printf("No layers found\n");
    free(buffer);
    return;
  }

  size_t layer_count =
      iree_exsleratev2_hal_exsleratev2_ExecutableDef_vec_len(layers);
  printf("Found %zu layers:\n", layer_count);

  for (size_t i = 0; i < layer_count; i++) {
    iree_exsleratev2_hal_exsleratev2_LayerDef_table_t layer =
        iree_exsleratev2_hal_exsleratev2_LayerDef_vec_at(layers, i);
    printf("Layer %zu:\n", i);

    iree_exsleratev2_hal_exsleratev2_RegisterValue_vec_t csr_configs =
        iree_exsleratev2_hal_exsleratev2_LayerDef_csr_configs(layer);
    if (csr_configs) {
      size_t csr_count =
          iree_exsleratev2_hal_exsleratev2_RegisterValue_vec_len(csr_configs);
      printf("  CSR Configs (%zu):\n", csr_count);
      for (size_t j = 0; j < csr_count; j++) {
        iree_exsleratev2_hal_exsleratev2_RegisterValue_table_t reg =
            iree_exsleratev2_hal_exsleratev2_RegisterValue_vec_at(csr_configs,
                                                                  j);
        uint8_t value_type =
            iree_exsleratev2_hal_exsleratev2_RegisterValue_value_type(reg);
        uint32_t literal_value =
            iree_exsleratev2_hal_exsleratev2_RegisterValue_literal_value(reg);
        uint32_t memref_id =
            iree_exsleratev2_hal_exsleratev2_RegisterValue_memrefdef_id(reg);

        printf("    %zu: type=%u, literal=%u, memref_id=%u\n", j, value_type,
               literal_value, memref_id);
      }
    }

    iree_exsleratev2_hal_exsleratev2_MemRefDef_vec_t mem_ref_defs =
        iree_exsleratev2_hal_exsleratev2_LayerDef_mem_ref_defs(layer);
    if (mem_ref_defs) {
      size_t memref_count =
          iree_exsleratev2_hal_exsleratev2_MemRefDef_vec_len(mem_ref_defs);
      printf("  MemRef Defs (%zu):\n", memref_count);
      for (size_t j = 0; j < memref_count; j++) {
        iree_exsleratev2_hal_exsleratev2_MemRefDef_table_t memref =
            iree_exsleratev2_hal_exsleratev2_MemRefDef_vec_at(mem_ref_defs, j);
        uint32_t id = iree_exsleratev2_hal_exsleratev2_MemRefDef_id(memref);
        int8_t data_type =
            iree_exsleratev2_hal_exsleratev2_MemRefDef_data_type(memref);

        uint32_t alignment =
            iree_exsleratev2_hal_exsleratev2_MemRefDef_alignment(memref);

        flatbuffers_int32_vec_t shape_ptr =
            iree_exsleratev2_hal_exsleratev2_MemRefDef_shape(memref);
        size_t shape = flatbuffers_int32_vec_len(shape_ptr);

        printf("  MemRef %zu: id=%u, type=%d, alignment=%u, shape=%zu\n", j, id,
               data_type, alignment, shape);

        printf("    Shape: [");
        for (size_t i = 0; i < shape; ++i) {
          printf("%d", shape_ptr[i]);
          if (i < shape - 1) {
            printf(", ");
          }
        }
        printf("]\n");

        printf("    %zu: id=%u, type=%d, alignment=%u\n", j, id, data_type,
               alignment);
      }
    }

    iree_exsleratev2_hal_exsleratev2_DataBufferDef_vec_t data_buffers =
        iree_exsleratev2_hal_exsleratev2_LayerDef_data_buffers(layer);
    if (data_buffers) {
      size_t buffer_count =
          iree_exsleratev2_hal_exsleratev2_DataBufferDef_vec_len(data_buffers);
      printf("  Data Buffers (%zu):\n", buffer_count);
      for (size_t j = 0; j < buffer_count; j++) {
        iree_exsleratev2_hal_exsleratev2_DataBufferDef_table_t data_buffer =
            iree_exsleratev2_hal_exsleratev2_DataBufferDef_vec_at(data_buffers,
                                                                  j);
        uint8_t category =
            iree_exsleratev2_hal_exsleratev2_DataBufferDef_category(
                data_buffer);

        flatbuffers_generic_t buffer =
            iree_exsleratev2_hal_exsleratev2_DataBufferDef_buffer(data_buffer);

        auto int8_buffer =
            (iree_exsleratev2_hal_exsleratev2_Int8Buffer_table_t)buffer;
        flatbuffers_int8_vec_t data =
            iree_exsleratev2_hal_exsleratev2_Int8Buffer_data(int8_buffer);

        printf("    %zu: category=%u\n", j, category);
        if (data) {
          printf("    Int8Buffer (%zu bytes):\n[",
                 flatbuffers_int8_vec_len(data));
          for (size_t i = 0; i < flatbuffers_int8_vec_len(data); i++) {
            printf("%d ", flatbuffers_int8_vec_at(data, i));
          }
        }
        printf("]\n");
      }
    }
  }

  free(buffer);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <path_to_flatbuffer_file>\n", argv[0]);
    return 1;
  }

  const char *filename = argv[1];
  deserializeFromSLFb(filename);

  return 0;
}