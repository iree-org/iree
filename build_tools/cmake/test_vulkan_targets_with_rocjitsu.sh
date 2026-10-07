#!/usr/bin/env bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs the regular build's Vulkan compute e2e and HAL CTS tests through RADV on
# simulated RDNA3/RDNA4 GPUs. Build all and iree-test-deps before running.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: test_vulkan_targets_with_rocjitsu.sh [BUILD_DIR] [TARGET ...]

BUILD_DIR defaults to IREE_BUILD_DIR or "build". Targets default to gfx1100
and gfx1201. Runs the Vulkan compute e2e and HAL CTS tests. The existing
SPIR-V test modules are shared between targets; RADV compiles them to the
simulated GPU's native instructions at runtime.

Requires Mesa RADV and rocjitsu with Vulkan compute support.
Set ROCJITSU_BIN and ROCJITSU_CONFIG_DIR for a source build, or ROCM_ROOT
for an initialized TheRock SDK that includes that support.

Configuration:
  VK_DRIVER_FILES             Path to a RADV ICD manifest (auto-detected).
  IREE_ROCJITSU_WORK_DIR       Artifact root (default: BUILD_DIR/rocjitsu).
  IREE_ROCJITSU_RUN_ID         Log run ID (default: timestamp and PID).
  CTEST_PARALLEL_LEVEL         Test concurrency (default: 8).
  IREE_ROCJITSU_TEST_TIMEOUT   Per-test default timeout in seconds (180).
  IREE_ROCJITSU_SESSION_TIMEOUT  Per-target watchdog in seconds (1800).
  IREE_ROCJITSU_TESTS_REGEX    Optional CTest test-name filter.

Example from the IREE repository root:
  ROCM_ROOT="$(rocm-sdk path --root)" \
    build_tools/cmake/test_vulkan_targets_with_rocjitsu.sh build gfx1100
EOF
}

BUILD_DIR="${1:-${IREE_BUILD_DIR:-build}}"
if [[ "${BUILD_DIR}" == --help || "${BUILD_DIR}" == -h ]]; then
  usage
  exit 0
fi
if (($#)); then shift; fi
if (($#)); then
  TARGETS=("$@")
else
  TARGETS=(gfx1100 gfx1201)
fi

declare -Ar CONFIGS=(
  [gfx1100]=gfx1100_w7900.json
  [gfx1201]=gfx1201_r9700.json
)
declare -Ar DEVICE_IDS=([gfx1100]=7448 [gfx1201]=7551)
for target in "${TARGETS[@]}"; do
  if [[ -z "${CONFIGS[${target}]+x}" ]]; then
    echo "error: unsupported target '${target}'" >&2
    exit 2
  fi
done

BUILD_DIR="$(realpath "${BUILD_DIR}")"
for setting in IREE_BUILD_TESTS IREE_TARGET_BACKEND_VULKAN_SPIRV IREE_HAL_DRIVER_VULKAN; do
  if ! grep -q "^${setting}:BOOL=ON$" "${BUILD_DIR}/CMakeCache.txt"; then
    echo "error: configure the IREE build with ${setting}=ON" >&2
    exit 2
  fi
done
if [[ -z "${ROCJITSU_BIN:-}" && -n "${ROCM_ROOT:-}" ]]; then
  ROCJITSU_BIN="${ROCM_ROOT}/bin/rocjitsu"
  if [[ ! -x "${ROCJITSU_BIN}" ]]; then ROCJITSU_BIN="${ROCM_ROOT}/bin/mirage"; fi
fi
: "${ROCJITSU_BIN:?Set ROCJITSU_BIN or ROCM_ROOT}"
ROCJITSU_BIN="$(realpath "${ROCJITSU_BIN}")"
ROCJITSU_CONFIG_DIR="${ROCJITSU_CONFIG_DIR:-${ROCM_ROOT:-}/share/rocjitsu/configs}"
if [[ ! -x "${ROCJITSU_BIN}" ]]; then
  echo "error: simulator executable not found: ${ROCJITSU_BIN}" >&2
  exit 2
fi

if [[ -z "${VK_DRIVER_FILES:-}" ]]; then
  for manifest in /usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
                  /usr/share/vulkan/icd.d/radeon_icd.json; do
    if [[ -f "${manifest}" ]]; then VK_DRIVER_FILES="${manifest}"; break; fi
  done
fi
: "${VK_DRIVER_FILES:?Install Mesa RADV or set VK_DRIVER_FILES}"
VK_DRIVER_FILES="$(realpath "${VK_DRIVER_FILES}")"
echo "RADV ICD: ${VK_DRIVER_FILES}"
IREE_RUN_MODULE="${BUILD_DIR}/tools/iree-run-module"
if [[ ! -x "${IREE_RUN_MODULE}" ]]; then
  echo "error: build all and iree-test-deps before running this script" >&2
  exit 2
fi
CTEST_BIN="$(command -v ctest)"
WORK_DIR="${IREE_ROCJITSU_WORK_DIR:-${BUILD_DIR}/rocjitsu}"
mkdir -p "${WORK_DIR}"
WORK_DIR="$(realpath "${WORK_DIR}")"
RUN_ID="${IREE_ROCJITSU_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"

# Use system Mesa/libdrm, and remove inherited hardware and loader selectors.
# Jammy's Vulkan loader requires the older VK_ICD_FILENAMES spelling too.
RUN_ENV=(env -u LD_LIBRARY_PATH -u LD_PRELOAD
  -u VK_ADD_DRIVER_FILES -u VK_INSTANCE_LAYERS -u VK_LOADER_DRIVERS_SELECT
  -u VK_LOADER_DRIVERS_DISABLE -u RADV_FORCE_FAMILY -u IREE_VULKAN_DISABLE
  "VK_DRIVER_FILES=${VK_DRIVER_FILES}" "VK_ICD_FILENAMES=${VK_DRIVER_FILES}" DRI_PRIME=)
CTEST_FILTER=(-L '^driver=vulkan$'
  -L '^iree/(tests/e2e/|hal/drivers/vulkan/cts$)'
  -LE '^requires-gpu-(nvidia|sm[0-9]+)$|^very-expensive$|^requires-multiple-devices$')
if [[ -n "${IREE_ROCJITSU_TESTS_REGEX:-}" ]]; then
  CTEST_FILTER+=(-R "${IREE_ROCJITSU_TESTS_REGEX}")
fi

"${ROCJITSU_BIN}" --version
status=0
for target in "${TARGETS[@]}"; do
  log_dir="${WORK_DIR}/logs/${RUN_ID}/vulkan/${target}"
  mkdir -p "${log_dir}"
  config="$(realpath "${ROCJITSU_CONFIG_DIR}/${CONFIGS[${target}]}")"
  echo "=== Vulkan e2e and HAL CTS tests: ${target} ==="
  if (
    # Keep socket paths short and isolate concurrent helper invocations.
    runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/iree-vulkan-${target}.XXXXXX")" || exit 1
    trap 'rm -rf -- "${runtime_dir}"' EXIT
    export ROCJITSU_RUNTIME_DIR="${runtime_dir}"
    export MIRAGE_RUNTIME="${runtime_dir}"
    simulator=("${ROCJITSU_BIN}")
    if [[ "$(basename "${ROCJITSU_BIN}")" == mirage ]]; then
      simulator+=(run --in-process)
    fi
    simulator+=(--config "${config}" --)
    run_simulator() {
      "${RUN_ENV[@]}" timeout --signal=TERM --kill-after=30 \
        "${IREE_ROCJITSU_SESSION_TIMEOUT:-1800}" "${simulator[@]}" "$@"
    }
    run_simulator "${IREE_RUN_MODULE}" --dump_devices=vulkan > "${log_dir}/device.txt" 2>&1 || {
      cat "${log_dir}/device.txt"
      exit 1
    }
    # Verify driver and device identity so missing simulation cannot give a
    # passing run on a software driver or an unrelated physical GPU.
    if [[ "$(sed -n 's/^vendor_id:[[:space:]]*//p' "${log_dir}/device.txt")" != 0x1002 ||
          "$(sed -n 's/^device_id:[[:space:]]*//p' "${log_dir}/device.txt")" != "0x${DEVICE_IDS[${target}]}" ||
          "$(sed -n 's/^driver:[[:space:]]*//p' "${log_dir}/device.txt")" != radv ]]; then
      cat "${log_dir}/device.txt"
      echo "error: expected exactly one RADV device with the simulated PCI ID" >&2
      exit 1
    fi
    grep -E '^(name|vendor_id|device_id|driver|driver_info):' "${log_dir}/device.txt"
    run_simulator "${CTEST_BIN}" --test-dir "${BUILD_DIR}" \
      --parallel "${CTEST_PARALLEL_LEVEL:-8}" \
      --timeout "${IREE_ROCJITSU_TEST_TIMEOUT:-180}" \
      --output-on-failure --no-tests=error \
      --output-junit "${log_dir}/ctest.xml" "${CTEST_FILTER[@]}"
  ) 2>&1 | tee "${log_dir}/ctest.log"; then
    echo "Passed: ${target}"
  else
    echo "Failed: ${target}" >&2
    status=1
  fi
done
echo "Logs: ${WORK_DIR}/logs/${RUN_ID}/vulkan"
exit "${status}"
