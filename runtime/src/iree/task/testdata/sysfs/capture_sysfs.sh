#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Captures the CPU topology of the current system as a single-file manifest.
#
# Manifest format: '# comment' lines plus one 'relative/path=content' line per
# sysfs file, rooted at /sys/devices/system. Only the files topology_sysfs.c
# reads are captured.
#
# Usage:
#   ./capture_sysfs.sh [manifest_file]                 # capture this machine
#   ./capture_sysfs.sh --expand <manifest> <dest_dir>  # manifest -> tree
#

set -e

SYSFS_ROOT="/sys/devices/system"

#===------------------------------------------------------------------------===
# --expand: manifest -> directory tree
#===------------------------------------------------------------------------===
if [ "$1" = "--expand" ]; then
  MANIFEST="$2"
  DEST="$3"
  if [ -z "${MANIFEST}" ] || [ -z "${DEST}" ]; then
    echo "usage: $0 --expand <manifest> <dest_dir>" >&2
    exit 1
  fi
  FILE_COUNT=0
  while IFS= read -r line; do
    case "${line}" in
      '' | '#'*) continue ;;
    esac
    path="${line%%=*}"
    value="${line#*=}"
    mkdir -p "${DEST}/$(dirname "${path}")"
    printf '%s\n' "${value}" > "${DEST}/${path}"
    FILE_COUNT=$((FILE_COUNT + 1))
  done < "${MANIFEST}"
  echo "Expanded ${FILE_COUNT} files from ${MANIFEST} to ${DEST}"
  echo ""
  exit 0
fi

#===------------------------------------------------------------------------===
# capture: this machine -> manifest
#===------------------------------------------------------------------------===
if [ -n "$1" ]; then
  OUT="$1"
else
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  HOSTNAME=$(hostname -s)
  OUT="captured_${HOSTNAME}_${TIMESTAMP}.sysfs.txt"
fi

echo "Capturing sysfs CPU topology to: ${OUT}"

{
  echo "# sysfs topology snapshot of $(hostname -s), captured $(date -u +%Y-%m-%d)."
  echo "# Format: one 'relative/path=content' line per sysfs file; expand with"
  echo "#   ./capture_sysfs.sh --expand <this file> <destdir>"
} > "${OUT}"

# Appends 'path=content' for one sysfs file if it exists. Sysfs values are
# single-line; the trailing newline is stripped here and re-added on expansion.
emit() {
  local path="$1"
  if [ -f "${SYSFS_ROOT}/${path}" ]; then
    printf '%s=%s\n' "${path}" "$(tr -d '\n' < "${SYSFS_ROOT}/${path}")" \
      >> "${OUT}"
  fi
}

# CPU enumeration (only what topology_sysfs.c reads: present, with kernel_max
# as its fallback).
emit "cpu/present"
emit "cpu/kernel_max"

# NUMA node hierarchy (node/online + per-node cpulist). This is the primary
# source of node identity for the backend.
emit "node/online"
for node_dir in "${SYSFS_ROOT}"/node/node[0-9]*; do
  if [ -d "${node_dir}" ]; then
    emit "node/$(basename "${node_dir}")/cpulist"
  fi
done

# Per-CPU information.
CPU_COUNT=0
for cpu_dir in "${SYSFS_ROOT}"/cpu/cpu[0-9]*; do
  if [ ! -d "${cpu_dir}" ]; then
    continue
  fi
  CPU_NAME=$(basename "${cpu_dir}")

  # cpu_capacity (ARM big.LITTLE).
  emit "cpu/${CPU_NAME}/cpu_capacity"

  # Topology. thread_siblings_list is captured because pre-5.3 kernels do not
  # expose core_cpus_list.
  for topo_file in core_id physical_package_id cluster_id core_cpus_list thread_siblings_list; do
    emit "cpu/${CPU_NAME}/topology/${topo_file}"
  done

  # Cache hierarchy.
  for cache_index_dir in "${cpu_dir}"/cache/index*; do
    if [ -d "${cache_index_dir}" ]; then
      INDEX_NAME=$(basename "${cache_index_dir}")
      for cache_file in type level size shared_cpu_list; do
        emit "cpu/${CPU_NAME}/cache/${INDEX_NAME}/${cache_file}"
      done
    fi
  done

  CPU_COUNT=$((CPU_COUNT + 1))
done

echo "Successfully captured ${CPU_COUNT} CPUs to ${OUT}"
echo ""
echo "To test, expand into a directory tree and point the tools at it:"
echo "  $0 --expand ${OUT} /tmp/$(basename "${OUT%.sysfs.txt}")"
