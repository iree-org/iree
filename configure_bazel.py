# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import platform
import os
import shutil
import subprocess
import sys


def detect_unix_platform_config(bazelrc):
    # This is hoaky. Ideally, bazel had any kind of rational way of selecting
    # options from within its environment (key word: "rational"), but sadly, it
    # is unintelligible to mere mortals. Why should a build system have a way for
    # people to condition their build options on what compiler they are using
    # (without descending down the hole of deciphering what a Bazel toolchain is)?
    # All I want to do is set a couple of project specific warning options!

    if platform.system() == "Darwin":
        print(f"build --config=macos_clang", file=bazelrc)
        print(f"build:release --config=macos_clang_release", file=bazelrc)
    else:
        # If the user specified a CXX environment var, bazel will later respect that,
        # so we just see if it says "clang".
        cxx = os.environ.get("CXX")
        cc = os.environ.get("CC")
        if (cxx is not None and cc is None) or (cxx is None and cc is not None):
            print(
                "WARNING: Only one of CXX or CC is set, which can confuse bazel. "
                "Recommend: set both appropriately (or none)"
            )
        if cc is not None and cxx is not None:
            # Persist the variables.
            print(f'build --action_env CC="{cc}"', file=bazelrc)
            print(f'build --action_env CXX="{cxx}"', file=bazelrc)
        else:
            print(
                "WARNING: CC and CXX are not set, which can cause mismatches between "
                "flag configurations and compiler. Recommend setting them explicitly."
            )

        if cxx is not None and "clang" in cxx:
            print(f"Choosing generic_clang config because CXX is set to clang ({cxx})")
            print(f"build --config=generic_clang", file=bazelrc)
            print(f"build:release --config=generic_clang_release", file=bazelrc)
        else:
            print(
                f"Choosing generic_gcc config by default because no CXX set or "
                f"not recognized as clang ({cxx})"
            )
            print(f"build --config=generic_gcc", file=bazelrc)
            print(f"build:release --config=generic_gcc_release", file=bazelrc)


def write_platform(bazelrc):
    if platform.system() == "Windows":
        print(f"build --config=msvc", file=bazelrc)
        print(f"build:release --config=msvc_release", file=bazelrc)
    else:
        detect_unix_platform_config(bazelrc)


def cmake_bool_is_true(value):
    """Check if a CMake-style bool value is true."""
    if not value:
        return False
    return value.upper() in ("ON", "YES", "TRUE", "Y", "1")


def detect_cuda_toolkit():
    """Check if CUDA toolkit is available.

    Follows MLIR's search order: CUDA_ROOT, CUDA_HOME, CUDA_PATH, then nvcc in PATH.
    """
    for env_var in ["CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"]:
        if os.environ.get(env_var):
            return True
    if shutil.which("nvcc"):
        return True
    return False


def detect_rocm_toolkit():
    """Check if ROCm toolkit is available.

    Follows MLIR's search order: ROCM_PATH, ROCM_ROOT, ROCM_HOME, then hipcc in PATH.
    """
    for env_var in ["ROCM_PATH", "ROCM_ROOT", "ROCM_HOME"]:
        if os.environ.get(env_var):
            return True
    if shutil.which("hipcc"):
        return True
    return False


def detect_submodule(submodule_path):
    """Check if a submodule is initialized by looking for a marker file.

    Args:
        submodule_path: Path relative to repo root (e.g., "third_party/torch-mlir")

    Returns:
        True if the submodule appears to be initialized.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check for CMakeLists.txt as a common marker that the submodule has content
    marker = os.path.join(script_dir, submodule_path, "CMakeLists.txt")
    return os.path.isfile(marker)


def get_hal_driver_defaults():
    """Get HAL driver defaults matching CMake option(IREE_HAL_DRIVER_*) definitions."""
    defaults_enabled = True  # Matches IREE_HAL_DRIVER_DEFAULTS in CMakeLists.txt

    return {
        "AMDGPU": False,
        "CUDA": False,
        "HIP": False,
        "LOCAL_SYNC": defaults_enabled,
        "LOCAL_TASK": defaults_enabled,
        "METAL": platform.system() == "Darwin" and defaults_enabled,
        "NULL": False,  # Special: OFF in tests, ON otherwise
        "VULKAN": defaults_enabled and platform.system() not in ("Android", "iOS"),
    }


def env_var_to_bazel_tag(name):
    """Convert env var name to Bazel tag format.

    Bazel tags use hyphens: local-task, vulkan-spirv
    Env vars use underscores: IREE_HAL_DRIVER_LOCAL_TASK
    """
    if name.startswith("IREE_HAL_DRIVER_"):
        tag_name = name[len("IREE_HAL_DRIVER_") :]
    else:
        tag_name = name
    return tag_name.lower().replace("_", "-")


def get_plugin_defaults():
    """Get compiler plugin defaults matching CMake option definitions.

    Returns a dict mapping plugin_id -> (default_enabled, env_var_name, can_build).
    The can_build field indicates whether the plugin can be built on this system
    (e.g., CUDA requires toolkit, Metal requires macOS, input plugins require submodules).
    """
    cuda_available = detect_cuda_toolkit()
    rocm_available = detect_rocm_toolkit()
    stablehlo_available = detect_submodule("third_party/stablehlo")
    torch_mlir_available = detect_submodule("third_party/torch-mlir")
    is_darwin = platform.system() == "Darwin"

    return {
        # Input plugins (require submodules)
        "input_stablehlo": (True, "IREE_INPUT_STABLEHLO", stablehlo_available),
        "input_tosa": (True, "IREE_INPUT_TOSA", True),  # Part of MLIR, no submodule
        "input_torch": (False, "IREE_INPUT_TORCH", torch_mlir_available),
        # Target plugins
        "hal_target_cuda": (False, "IREE_TARGET_BACKEND_CUDA", cuda_available),
        "hal_target_llvm_cpu": (True, "IREE_TARGET_BACKEND_LLVM_CPU", True),
        "hal_target_local": (True, "IREE_TARGET_BACKEND_LOCAL", True),
        "hal_target_metal_spirv": (
            is_darwin,
            "IREE_TARGET_BACKEND_METAL_SPIRV",
            is_darwin,
        ),
        "hal_target_rocm": (False, "IREE_TARGET_BACKEND_ROCM", rocm_available),
        "hal_target_vmvx": (True, "IREE_TARGET_BACKEND_VMVX", True),
        "hal_target_vulkan_spirv": (True, "IREE_TARGET_BACKEND_VULKAN_SPIRV", True),
        # Sample plugins (always buildable)
        "example": (True, None, True),
        "simple_io_sample": (True, None, True),
    }


def parse_plugin_spec(spec, plugin_defaults):
    """Parse a plugin specification that may include 'all' and exclusions.

    Supports:
      - "all" - all plugins that can be built on this system
      - "all,-plugin1,-plugin2" - all except specified plugins
      - "plugin1,plugin2" - explicit list

    Returns a tuple (plugin_list, used_all) where used_all indicates whether
    "all" expansion was performed, or (None, False) if no spec provided.
    """
    if not spec:
        return None, False

    parts = [p.strip().lower() for p in spec.split(",")]
    if parts[0] == "all":
        # Start with all buildable plugins
        enabled = set(
            plugin_id
            for plugin_id, (_, _, can_build) in plugin_defaults.items()
            if can_build
        )

        # Process exclusions
        for part in parts[1:]:
            if part.startswith("-"):
                plugin_to_exclude = part[1:]
                if plugin_to_exclude in enabled:
                    enabled.discard(plugin_to_exclude)
                elif plugin_to_exclude not in plugin_defaults:
                    print(f"WARNING: Unknown plugin in exclusion: {plugin_to_exclude}")

        return sorted(enabled), True
    else:
        # Explicit list - validate plugins exist and can be built
        validated = []
        for plugin_id in parts:
            if plugin_id not in plugin_defaults:
                print(f"WARNING: Unknown plugin: {plugin_id}")
                continue
            _, _, can_build = plugin_defaults[plugin_id]
            if not can_build:
                print(
                    f"ERROR: Plugin '{plugin_id}' requested but cannot be built "
                    f"(missing toolkit/submodule). Either install prerequisites or "
                    f"remove from IREE_COMPILER_PLUGINS."
                )
                sys.exit(1)
            validated.append(plugin_id)
        return validated, False


def write_iree_plugin_options(bazelrc):
    """Write compiler plugin configuration to bazelrc."""
    plugin_defaults = get_plugin_defaults()

    # Check for IREE_COMPILER_PLUGINS env var with "all" support
    plugins_spec = os.environ.get("IREE_COMPILER_PLUGINS")
    parsed_plugins, used_all = parse_plugin_spec(plugins_spec, plugin_defaults)

    if parsed_plugins is not None:
        # Explicit specification via IREE_COMPILER_PLUGINS
        enabled_plugins = parsed_plugins
        if used_all:
            print(
                f"IREE_COMPILER_PLUGINS=all resolved to: {', '.join(enabled_plugins)}"
            )
        else:
            print(f"IREE_COMPILER_PLUGINS set to: {', '.join(enabled_plugins)}")
    else:
        # Standard per-plugin env var processing
        enabled_plugins = []
        for plugin_id, (default, env_var, can_build) in plugin_defaults.items():
            if env_var:
                env_value = os.environ.get(env_var)
                enabled = (
                    cmake_bool_is_true(env_value) if env_value is not None else default
                )
            else:
                enabled = default

            if enabled:
                if not can_build:
                    print(
                        f"WARNING: {plugin_id} enabled but toolkit not detected, skipping"
                    )
                else:
                    enabled_plugins.append(plugin_id)

    # Write --iree_compiler_plugins flag (controls what gets built and linked)
    # Always emit the flag, even for empty list, so Bazel doesn't fall back to defaults.
    print(f'build --iree_compiler_plugins={",".join(enabled_plugins)}', file=bazelrc)

    # If torch is enabled, set IREE_INPUT_TORCH for the repository rule.
    # This uses --repo_env (visible to repository rules) not --action_env.
    if "input_torch" in enabled_plugins:
        print("common --repo_env=IREE_INPUT_TORCH=ON", file=bazelrc)


def write_iree_hal_driver_options(bazelrc):
    """Write HAL driver configuration to bazelrc."""

    # Get defaults matching CMake
    hal_drivers = get_hal_driver_defaults()

    # Apply environment overrides
    enabled_drivers = []
    for driver, default in hal_drivers.items():
        env_var = f"IREE_HAL_DRIVER_{driver}"
        env_value = os.environ.get(env_var)
        enabled = cmake_bool_is_true(env_value) if env_value is not None else default

        if enabled:
            enabled_drivers.append(env_var_to_bazel_tag(env_var))

    # Write --iree_runtime_drivers flag (controls what gets built and linked)
    if enabled_drivers:
        print(f'build --iree_runtime_drivers={",".join(enabled_drivers)}', file=bazelrc)
        print(f'test --iree_runtime_drivers={",".join(enabled_drivers)}', file=bazelrc)


if len(sys.argv) > 1:
    local_bazelrc = sys.argv[1]
else:
    local_bazelrc = os.path.join(os.path.dirname(__file__), "configured.bazelrc")
with open(local_bazelrc, "wt") as bazelrc:
    write_platform(bazelrc)
    write_iree_hal_driver_options(bazelrc)
    write_iree_plugin_options(bazelrc)

print("Wrote", local_bazelrc)
