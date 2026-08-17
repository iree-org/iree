#!/usr/bin/env bash
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Captures /sys/devices/system/cpu/ topology information for testing.
# This script creates a snapshot of the current system's CPU topology that can
# be used to test topology_sysfs.c without requiring physical hardware.
#
# Usage:
#   ./capture_sysfs.sh [destination_dir]
#
# If destination_dir is not provided, it defaults to a timestamped directory.

set -e

# Determine destination directory.
if [ -n "$1" ]; then
  DEST="$1"
else
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  HOSTNAME=$(hostname -s)
  DEST="captured_${HOSTNAME}_${TIMESTAMP}"
fi

DEST_CPU="${DEST}/cpu"
mkdir -p "${DEST_CPU}"

echo "Capturing sysfs CPU topology to: ${DEST}"

# Copy top-level CPU files.
for file in present possible online offline kernel_max sched_isolated isolated; do
  SRC="/sys/devices/system/cpu/${file}"
  if [ -f "${SRC}" ]; then
    cp "${SRC}" "${DEST_CPU}/" 2>/dev/null || true
  fi
done

# Capture per-CPU information.
CPU_COUNT=0
for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
  if [ ! -d "${cpu_dir}" ]; then
    continue
  fi

  CPU_NAME=$(basename "${cpu_dir}")
  DEST_CPU_DIR="${DEST_CPU}/${CPU_NAME}"
  mkdir -p "${DEST_CPU_DIR}"

  # Copy cpu_capacity (ARM big.LITTLE).
  if [ -f "${cpu_dir}/cpu_capacity" ]; then
    cp "${cpu_dir}/cpu_capacity" "${DEST_CPU_DIR}/"
  fi

  # Copy topology information.
  if [ -d "${cpu_dir}/topology" ]; then
    DEST_TOPO="${DEST_CPU_DIR}/topology"
    mkdir -p "${DEST_TOPO}"
    for topo_file in core_id physical_package_id cluster_id core_cpus_list thread_siblings_list; do
      SRC_FILE="${cpu_dir}/topology/${topo_file}"
      if [ -f "${SRC_FILE}" ]; then
        cp "${SRC_FILE}" "${DEST_TOPO}/"
      fi
    done
  fi

  # Copy cache hierarchy.
  if [ -d "${cpu_dir}/cache" ]; then
    DEST_CACHE="${DEST_CPU_DIR}/cache"
    mkdir -p "${DEST_CACHE}"

    for cache_index_dir in "${cpu_dir}"/cache/index*; do
      if [ ! -d "${cache_index_dir}" ]; then
        continue
      fi

      INDEX_NAME=$(basename "${cache_index_dir}")
      DEST_INDEX="${DEST_CACHE}/${INDEX_NAME}"
      mkdir -p "${DEST_INDEX}"

      for cache_file in type level size coherency_line_size number_of_sets physical_line_partition shared_cpu_list; do
        SRC_FILE="${cache_index_dir}/${cache_file}"
        if [ -f "${SRC_FILE}" ]; then
          cp "${SRC_FILE}" "${DEST_INDEX}/"
        fi
      done
    done
  fi

  CPU_COUNT=$((CPU_COUNT + 1))
done

echo "Successfully captured ${CPU_COUNT} CPUs to ${DEST}"

# Capture NUMA node topology (sibling of cpu/ under /sys/devices/system).
DEST_NODE="${DEST}/node"
if [ -d /sys/devices/system/node ]; then
  mkdir -p "${DEST_NODE}"
  for file in online possible has_cpu; do
    SRC="/sys/devices/system/node/${file}"
    if [ -f "${SRC}" ]; then
      cp "${SRC}" "${DEST_NODE}/" 2>/dev/null || true
    fi
  done
  NODE_COUNT=0
  for node_dir in /sys/devices/system/node/node[0-9]*; do
    if [ ! -d "${node_dir}" ]; then
      continue
    fi
    NODE_NAME=$(basename "${node_dir}")
    DEST_NODE_DIR="${DEST_NODE}/${NODE_NAME}"
    mkdir -p "${DEST_NODE_DIR}"
    for node_file in cpulist cpumap distance; do
      SRC_FILE="${node_dir}/${node_file}"
      if [ -f "${SRC_FILE}" ]; then
        cp "${SRC_FILE}" "${DEST_NODE_DIR}/"
      fi
    done
    NODE_COUNT=$((NODE_COUNT + 1))
  done
  echo "Successfully captured ${NODE_COUNT} NUMA nodes to ${DEST_NODE}"
else
  echo "No /sys/devices/system/node present; skipped NUMA capture"
fi

echo ""
echo "To test with this snapshot:"
echo "  ./build/tools/iree-run-module \\"
echo "    --task_topology_sysfs_root=$(realpath "${DEST}") \\"
echo "    --dump_task_topologies"
