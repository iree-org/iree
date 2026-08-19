# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# NOTE: this file is not bazel-to-cmake generated.
# Keep the manifest list in sync with BUILD.bazel.

# Mirrors :expanded_sysfs_snapshots. The snapshots are only consumed by
# iree/task topology_test which only exercises them on Linux.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_IREE_TASK_SYSFS_MANIFEST_NAMES
      "arm64_pixel6_tensor"
      "synthetic_big_little"
      "synthetic_dual_socket"
      "synthetic_dual_socket_no_siblings"
      "synthetic_fragmented_cpulist"
      "synthetic_invalid_package"
      "synthetic_memory_only_node"
      "synthetic_meteor_lake"
      "synthetic_raptor_lake"
      "synthetic_snc"
      "synthetic_sparse_node_ids"
      "synthetic_too_many_nodes"
      "x86_64_ryzen9_7950x")
  set(_IREE_TASK_SYSFS_MANIFESTS)
  foreach(_IREE_TASK_SYSFS_NAME ${_IREE_TASK_SYSFS_MANIFEST_NAMES})
    list(APPEND _IREE_TASK_SYSFS_MANIFESTS
         "${CMAKE_CURRENT_SOURCE_DIR}/${_IREE_TASK_SYSFS_NAME}.sysfs.txt")
  endforeach()
  set(_IREE_TASK_SYSFS_EXPAND_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/expanded_sysfs_snapshots")
  set(_IREE_TASK_SYSFS_STAMP "${_IREE_TASK_SYSFS_EXPAND_DIR}.stamp")
  set(_IREE_TASK_SYSFS_EXPAND_COMMANDS)
  foreach(_IREE_TASK_SYSFS_MANIFEST ${_IREE_TASK_SYSFS_MANIFESTS})
    get_filename_component(_IREE_TASK_SYSFS_BASENAME
                           "${_IREE_TASK_SYSFS_MANIFEST}" NAME)
    string(REPLACE ".sysfs.txt" "" _IREE_TASK_SYSFS_NAME
           "${_IREE_TASK_SYSFS_BASENAME}")
    list(APPEND _IREE_TASK_SYSFS_EXPAND_COMMANDS
         COMMAND bash
                 "${CMAKE_CURRENT_SOURCE_DIR}/capture_sysfs.sh"
                 --expand "${_IREE_TASK_SYSFS_MANIFEST}"
                 "${_IREE_TASK_SYSFS_EXPAND_DIR}/${_IREE_TASK_SYSFS_NAME}")
  endforeach()
  add_custom_command(
    OUTPUT "${_IREE_TASK_SYSFS_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${_IREE_TASK_SYSFS_EXPAND_DIR}"
    ${_IREE_TASK_SYSFS_EXPAND_COMMANDS}
    COMMAND ${CMAKE_COMMAND} -E touch "${_IREE_TASK_SYSFS_STAMP}"
    DEPENDS ${_IREE_TASK_SYSFS_MANIFESTS}
            "${CMAKE_CURRENT_SOURCE_DIR}/capture_sysfs.sh"
    COMMENT "Expanding sysfs snapshot manifests")
  add_custom_target(iree_runtime_task_expanded_sysfs_snapshots ALL
                    DEPENDS "${_IREE_TASK_SYSFS_STAMP}")
endif()
