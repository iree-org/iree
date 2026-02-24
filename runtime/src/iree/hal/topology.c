// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/topology.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_topology_t formatting
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_topology_edge_format(iree_hal_topology_edge_t edge,
                                            iree_string_builder_t* builder) {
  static const iree_bitfield_string_mapping_t interop_mode_mappings[] = {
      {IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE, IREE_SVL("NATIVE")},
      {IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT, IREE_SVL("IMPORT")},
      {IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY, IREE_SVL("COPY")},
      {IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE, IREE_SVL("NONE")},
  };

  static const iree_bitfield_string_mapping_t link_class_mappings[] = {
      {IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE, IREE_SVL("SAME_DIE")},
      {IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF, IREE_SVL("NVLINK/IF")},
      {IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT, IREE_SVL("PCIE")},
      {IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_CROSS_ROOT, IREE_SVL("PCIE_CROSS")},
      {IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED, IREE_SVL("HOST")},
      {IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC, IREE_SVL("FABRIC")},
      {IREE_HAL_TOPOLOGY_LINK_CLASS_OTHER, IREE_SVL("OTHER")},
      {IREE_HAL_TOPOLOGY_LINK_CLASS_ISOLATED, IREE_SVL("ISOLATED")},
  };

  // Format wait mode.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "wait="));
  IREE_RETURN_IF_ERROR(iree_bitfield_format(
      iree_hal_topology_edge_wait_mode(edge.lo),
      IREE_ARRAYSIZE(interop_mode_mappings), interop_mode_mappings, builder));

  // Format signal mode.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " signal="));
  IREE_RETURN_IF_ERROR(iree_bitfield_format(
      iree_hal_topology_edge_signal_mode(edge.lo),
      IREE_ARRAYSIZE(interop_mode_mappings), interop_mode_mappings, builder));

  // Format link class.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " link="));
  IREE_RETURN_IF_ERROR(iree_bitfield_format(
      iree_hal_topology_edge_link_class(edge.lo),
      IREE_ARRAYSIZE(link_class_mappings), link_class_mappings, builder));

  // Format costs.
  uint8_t wait_cost = iree_hal_topology_edge_wait_cost(edge.lo);
  uint8_t copy_cost = iree_hal_topology_edge_copy_cost(edge.lo);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, " wait_cost=%u copy_cost=%u", wait_cost, copy_cost));

  return iree_ok_status();
}

iree_status_t iree_hal_topology_dump_matrix(const iree_hal_topology_t* topology,
                                            iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "Topology Matrix (%u devices):\n", topology->device_count));

  // Header row.
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("      ")));
  for (uint32_t j = 0; j < topology->device_count; ++j) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "  D%u  ", j));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("\n")));

  // Matrix rows.
  for (uint32_t i = 0; i < topology->device_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "D%u:  ", i));

    for (uint32_t j = 0; j < topology->device_count; ++j) {
      iree_hal_topology_edge_t edge =
          iree_hal_topology_query_edge(topology, i, j);

      // Simplified display showing wait mode.
      char mode_char = '?';
      switch (iree_hal_topology_edge_wait_mode(edge.lo)) {
        case IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE:
          mode_char = 'N';
          break;
        case IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT:
          mode_char = 'I';
          break;
        case IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY:
          mode_char = 'C';
          break;
        case IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE:
          mode_char = '-';
          break;
      }

      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(builder, "  %c   ", mode_char));
    }

    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("\n")));
  }

  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, IREE_SV("\nLegend: N=Native, I=Import, C=Copy, -=None\n")));

  return iree_ok_status();
}
