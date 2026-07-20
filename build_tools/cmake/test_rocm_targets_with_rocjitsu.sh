#!/usr/bin/env bash

# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Reconfigures a regular IREE build for one or more ROCm test targets and runs
# its HIP e2e tests under rocjitsu. Expects to be run from the root of the IREE
# repository after build_tools/cmake/build_all.sh has populated BUILD_DIR.
#
# Requires an initialized ROCm SDK containing rocjitsu. Install the
# SDK separately and set ROCM_ROOT; this script does not install dependencies.
# Set IREE_ROCM_TEST_TARGET_CHIP to the first target when running build_all.sh
# to include its test modules in the initial build.

set -euo pipefail

SOURCE_DIR="$(pwd)"
BUILD_DIR="${1:-${IREE_BUILD_DIR:-build}}"
if (($#)); then
  shift
fi

if [[ ! -f "${SOURCE_DIR}/CMakeLists.txt" ||
      ! -x "${SOURCE_DIR}/build_tools/cmake/build_all.sh" ]]; then
  echo "error: run this script from the root of the IREE repository" >&2
  exit 2
fi

readonly -a ALL_TARGETS=(gfx942 gfx950 gfx1100 gfx1201 gfx1250)
declare -Ar ROCJITSU_CONFIGS=(
  [gfx942]="gfx942_cdna3.json"
  [gfx950]="gfx950_mi355x.json"
  [gfx1100]="gfx1100_w7900.json"
  [gfx1201]="gfx1201_r9700.json"
  [gfx1250]="gfx1250_mi455x.json"
)
declare -Ar INCOMPATIBLE_GPU_LABELS=(
  [gfx942]='^requires-gpu-(cdna4|rdna3|rdna4|gfx1250)$'
  [gfx950]='^requires-gpu-(cdna3|rdna3|rdna4|gfx1250)$'
  [gfx1100]='^requires-gpu-(cdna3|cdna4|rdna4|gfx1250)$'
  [gfx1201]='^requires-gpu-(cdna3|cdna4|rdna3|gfx1250)$'
  [gfx1250]='^requires-gpu-(cdna3|cdna4|rdna3|rdna4|wave64)$'
)

usage() {
  cat <<'EOF'
Usage: build_tools/cmake/test_rocm_targets_with_rocjitsu.sh [BUILD_DIR] [TARGET ...]

Reconfigures an existing regular IREE build and runs its ROCm e2e tests under
rocjitsu. Supported targets:
  gfx942 gfx950 gfx1100 gfx1201 gfx1250

BUILD_DIR defaults to IREE_BUILD_DIR or "build". With no target arguments, all
supported targets are built and tested sequentially in the same build tree.

Requires an initialized ROCm SDK containing rocjitsu. Install the
SDK separately, run rocm-sdk init, and set ROCM_ROOT to its root directory.
Set IREE_ROCM_TEST_TARGET_CHIP to the first target when running build_all.sh
to include its test modules in the initial build.

Runs one parallel CTest invocation per target. CTest prints test progress and
output on failure; the target's log directory contains the log and JUnit report.
Each test process uses an independent simulator so concurrent tests do not
contend for a shared emulated GPU.

Configuration:
  IREE_ROCJITSU_WORK_DIR     Logs and simulator runtime state.
  IREE_ROCJITSU_RUN_ID       Override the log and simulator runtime run ID.
  ROCM_ROOT                  Required: root of an initialized ROCm SDK.
  ROCJITSU_BIN               Override the rocjitsu CLI executable.
  ROCJITSU_CONFIG_DIR        Override the rocjitsu config directory.
  CTEST_PARALLEL_LEVEL       Test concurrency (default: 8).
  IREE_ROCJITSU_TEST_TIMEOUT Default timeout in seconds for tests without an
                             explicit CMake TIMEOUT value (default: 180).
  IREE_ROCJITSU_SESSION_TIMEOUT
                             Whole-session watchdog in seconds (default: 1800).
  IREE_ROCJITSU_TESTS_REGEX  Include only tests matching this CTest regex.
  IREE_ROCJITSU_EXCLUDE_TESTS_REGEX
                             Exclude tests matching this CTest regex.
  IREE_ROCJITSU_EXCLUDE_LABEL_REGEX
                             Exclude tests with labels matching this regex
                             (default: ^very-expensive$; use ^$ to disable).

Examples:
  export ROCM_ROOT="$(rocm-sdk path --root)"
  build_tools/cmake/test_rocm_targets_with_rocjitsu.sh build gfx1250
  build_tools/cmake/test_rocm_targets_with_rocjitsu.sh build
  ROCM_ROOT=/path/to/rocm-sdk \
    build_tools/cmake/test_rocm_targets_with_rocjitsu.sh build gfx942 gfx950
EOF
}

if [[ "${BUILD_DIR}" == "--help" || "${BUILD_DIR}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ -z "${ROCM_ROOT:-}" ]]; then
  echo "error: set ROCM_ROOT to an initialized ROCm SDK containing rocjitsu" >&2
  exit 2
fi

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  echo "error: ${BUILD_DIR} is not a configured IREE build directory" >&2
  echo "Run build_tools/cmake/build_all.sh ${BUILD_DIR} first." >&2
  exit 2
fi
BUILD_DIR="$(realpath "${BUILD_DIR}")"

WORK_DIR="${IREE_ROCJITSU_WORK_DIR:-${BUILD_DIR}/rocjitsu}"
CTEST_PARALLEL_LEVEL="${CTEST_PARALLEL_LEVEL:-8}"
TEST_TIMEOUT="${IREE_ROCJITSU_TEST_TIMEOUT:-180}"
SESSION_TIMEOUT="${IREE_ROCJITSU_SESSION_TIMEOUT:-1800}"
EXCLUDE_LABEL_REGEX="${IREE_ROCJITSU_EXCLUDE_LABEL_REGEX:-^very-expensive$}"
RUN_ID="${IREE_ROCJITSU_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"

if (($#)); then
  TARGETS=("$@")
else
  TARGETS=("${ALL_TARGETS[@]}")
fi

for target in "${TARGETS[@]}"; do
  if [[ -z "${ROCJITSU_CONFIGS[${target}]+x}" ]]; then
    echo "error: unsupported target '${target}'" >&2
    usage >&2
    exit 2
  fi
done

declare -a CTEST_BASE_FILTER_ARGS=(
  -L '^driver=hip$'
  -L '^iree/tests/e2e/'
)
if [[ -n "${IREE_ROCJITSU_TESTS_REGEX:-}" ]]; then
  CTEST_BASE_FILTER_ARGS+=(-R "${IREE_ROCJITSU_TESTS_REGEX}")
fi
if [[ -n "${IREE_ROCJITSU_EXCLUDE_TESTS_REGEX:-}" ]]; then
  CTEST_BASE_FILTER_ARGS+=(-E "${IREE_ROCJITSU_EXCLUDE_TESTS_REGEX}")
fi

require_cache_setting() {
  local setting="$1"
  if ! grep -q "^${setting}:BOOL=ON$" "${BUILD_DIR}/CMakeCache.txt"; then
    echo "error: regular build must have ${setting}=ON" >&2
    exit 2
  fi
}

require_cache_setting IREE_BUILD_TESTS
require_cache_setting IREE_BUILD_COMPILER
require_cache_setting IREE_HAL_DRIVER_HIP
require_cache_setting IREE_TARGET_BACKEND_ROCM
if ! grep -q '^CMAKE_GENERATOR:INTERNAL=Ninja$' "${BUILD_DIR}/CMakeCache.txt"; then
  echo "error: this script expects the regular CI Ninja build" >&2
  exit 2
fi

mkdir -p "${WORK_DIR}"

ROCM_ROOT="$(realpath "${ROCM_ROOT}")"
export ROCM_ROOT

if [[ -z "${ROCJITSU_BIN:-}" ]]; then
  if [[ -x "${ROCM_ROOT}/bin/rocjitsu" ]]; then
    ROCJITSU_BIN="${ROCM_ROOT}/bin/rocjitsu"
  else
    ROCJITSU_BIN="${ROCM_ROOT}/bin/mirage"
  fi
fi
ROCJITSU_CONFIG_DIR="${ROCJITSU_CONFIG_DIR:-${ROCM_ROOT}/share/rocjitsu/configs}"
if [[ ! -x "${ROCJITSU_BIN}" ]]; then
  echo "error: rocjitsu CLI not found at ${ROCJITSU_BIN}" >&2
  echo "Install a nightly that includes a simulator CLI or set ROCJITSU_BIN." >&2
  exit 2
fi
case "$(basename "${ROCJITSU_BIN}")" in
  mirage) ROCJITSU_CLI_KIND="session" ;;
  *) ROCJITSU_CLI_KIND="legacy" ;;
esac
if [[ ! -f "${ROCM_ROOT}/lib/libamdhip64.so" ]]; then
  echo "error: HIP runtime not found under ${ROCM_ROOT}/lib" >&2
  exit 2
fi

echo "IREE build:      ${BUILD_DIR}"
echo "ROCm SDK root:   ${ROCM_ROOT}"
echo "Simulator CLI:   ${ROCJITSU_BIN} ($("${ROCJITSU_BIN}" --version))"
echo "Targets:         ${TARGETS[*]}"

declare -a passed_targets=()
declare -a build_failed_targets=()
declare -a test_failed_targets=()

for target in "${TARGETS[@]}"; do
  config_path="${ROCJITSU_CONFIG_DIR}/${ROCJITSU_CONFIGS[${target}]}"
  runtime_dir="${WORK_DIR}/runtime/${RUN_ID}/${target}"
  log_dir="${WORK_DIR}/logs/${RUN_ID}/${target}"
  mkdir -p "${runtime_dir}" "${log_dir}"

  label_exclude_regex="${INCOMPATIBLE_GPU_LABELS[${target}]}"
  if [[ -n "${EXCLUDE_LABEL_REGEX}" ]]; then
    label_exclude_regex="(${label_exclude_regex})|(${EXCLUDE_LABEL_REGEX})"
  fi
  target_ctest_filter_args=(
    "${CTEST_BASE_FILTER_ARGS[@]}"
    -LE "${label_exclude_regex}"
  )

  if [[ ! -f "${config_path}" ]]; then
    echo "error: rocjitsu config not found: ${config_path}" >&2
    build_failed_targets+=("${target} (missing config)")
    continue
  fi

  echo
  echo "=== Retargeting regular IREE build for ${target} ==="
  # Bash disables errexit inside conditional subshells, so propagate failures
  # explicitly instead of letting the final CTest listing mask a failed build.
  if (
    cmake \
      -S "${SOURCE_DIR}" \
      -B "${BUILD_DIR}" \
      "-DIREE_NATIVE_TEST_TIMEOUT_DEFAULT=${TEST_TIMEOUT}" \
      "-DIREE_ROCM_TEST_TARGET_CHIP=${target}" || exit 1
    cmake --build "${BUILD_DIR}" --target iree-test-deps -- -k 0 || exit 1
    ctest \
      --test-dir "${BUILD_DIR}" \
      --show-only=human \
      "${target_ctest_filter_args[@]}"
  ) 2>&1 | tee "${log_dir}/build.log"; then
    echo "Built ${target} ROCm test dependencies"
  else
    build_failed_targets+=("${target}")
    continue
  fi

  echo
  if (
    export IREE_HIP_DYLIB_PATH="${ROCM_ROOT}/lib"
    export LD_LIBRARY_PATH="${ROCM_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    ctest_bin="$(command -v ctest)" || exit 1
    ctest_command=(
      "${ctest_bin}"
      --test-dir "${BUILD_DIR}"
      --parallel "${CTEST_PARALLEL_LEVEL}"
      --timeout "${TEST_TIMEOUT}"
      --output-on-failure
      --no-tests=error
      --output-junit "${log_dir}/ctest.xml"
      "${target_ctest_filter_args[@]}"
    )
    if [[ "${ROCJITSU_CLI_KIND}" == "session" ]]; then
      session_runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/iree-rocjitsu-${target}.XXXXXX")" || exit 1
      cleanup_session_runtime() {
        rm -rf -- "${session_runtime_dir}" || exit 1
      }
      trap cleanup_session_runtime EXIT
      export MIRAGE_RUNTIME="${session_runtime_dir}"
      # Independent tests need no shared GPU state. Give each CTest child its
      # own simulator instead of contending for a single daemon's workers.
      timeout \
        --signal=TERM \
        --kill-after=30 \
        "${SESSION_TIMEOUT}" \
        "${ROCJITSU_BIN}" run \
        --in-process \
        --config "${config_path}" \
        --workdir "${BUILD_DIR}" \
        --env "IREE_HIP_DYLIB_PATH=${IREE_HIP_DYLIB_PATH}" \
        --env "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
        -- "${ctest_command[@]}"
    else
      export ROCJITSU_RUNTIME_DIR="${runtime_dir}"
      "${ROCJITSU_BIN}" --config "${config_path}" -- "${ctest_command[@]}"
    fi
  ) 2>&1 | tee "${log_dir}/ctest.log"; then
    passed_targets+=("${target}")
  else
    test_failed_targets+=("${target}")
  fi
done

echo
echo "=== Rocjitsu ROCm e2e summary ==="
echo "Passed:       ${passed_targets[*]:-none}"
echo "Build failed: ${build_failed_targets[*]:-none}"
echo "Test failed:  ${test_failed_targets[*]:-none}"
echo "Logs:         ${WORK_DIR}/logs/${RUN_ID}"

if ((${#build_failed_targets[@]} || ${#test_failed_targets[@]})); then
  exit 1
fi
