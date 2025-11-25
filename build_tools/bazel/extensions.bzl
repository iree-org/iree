# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bzlmod extension for IREE repository rules."""

load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository", "new_local_repository")
load("//build_tools/bazel:workspace.bzl", "cuda_auto_configure")

# We need to import llvm_configure from a path relative to llvm-raw
# but llvm-raw doesn't exist yet. So we use a repository rule wrapper.
def _llvm_configure_impl(repository_ctx):
    """Repository rule that delegates to LLVM's llvm_configure."""
    # Get the path to llvm-project
    llvm_path = repository_ctx.path(Label("@llvm-raw//:WORKSPACE")).dirname

    # Run the overlay script
    src_path = llvm_path
    bazel_path = src_path.get_child("utils").get_child("bazel")
    overlay_path = bazel_path.get_child("llvm-project-overlay")
    script_path = bazel_path.get_child("overlay_directories.py")

    python_bin = repository_ctx.which("python3")
    if not python_bin:
        python_bin = repository_ctx.which("python")

    if not python_bin:
        fail("Failed to find python3 binary")

    cmd = [
        python_bin,
        script_path,
        "--src",
        src_path,
        "--overlay",
        overlay_path,
        "--target",
        ".",
    ]
    exec_result = repository_ctx.execute(cmd, timeout = 20)

    if exec_result.return_code != 0:
        fail(("Failed to execute overlay script: '{cmd}'\n" +
              "Exited with code {return_code}\n" +
              "stdout:\n{stdout}\n" +
              "stderr:\n{stderr}\n").format(
            cmd = " ".join([str(arg) for arg in cmd]),
            return_code = exec_result.return_code,
            stdout = exec_result.stdout,
            stderr = exec_result.stderr,
        ))

    # Extract CMake settings for vars.bzl
    llvm_cmake_path = repository_ctx.path(llvm_path.get_child("llvm").get_child("CMakeLists.txt"))
    version_cmake_path = repository_ctx.path(llvm_path.get_child("cmake").get_child("Modules").get_child("LLVMVersion.cmake"))

    c = {
        "CMAKE_CXX_STANDARD": None,
        "LLVM_VERSION_MAJOR": None,
        "LLVM_VERSION_MINOR": None,
        "LLVM_VERSION_PATCH": None,
        "LLVM_VERSION_SUFFIX": None,
    }

    for cmake_file in [llvm_cmake_path, version_cmake_path]:
        for line in repository_ctx.read(cmake_file).splitlines():
            setfoo = line.partition("(")
            if setfoo[1] != "(":
                continue
            if setfoo[0].strip().lower() != "set":
                continue
            kv = setfoo[2].strip()
            i = kv.find(" ")
            if i < 0:
                continue
            k = kv[:i]
            if k == "LLVM_REQUIRED_CXX_STANDARD":
                k = "CMAKE_CXX_STANDARD"
                c[k] = None
            if k not in c:
                continue
            if c[k] != None:
                continue
            v = kv[i:].strip().partition(")")[0].partition(" ")[0]
            c[k] = v

    c["LLVM_VERSION"] = "{}.{}.{}".format(
        c["LLVM_VERSION_MAJOR"],
        c["LLVM_VERSION_MINOR"],
        c["LLVM_VERSION_PATCH"],
    )
    c["PACKAGE_VERSION"] = "{}.{}.{}{}".format(
        c["LLVM_VERSION_MAJOR"],
        c["LLVM_VERSION_MINOR"],
        c["LLVM_VERSION_PATCH"],
        c["LLVM_VERSION_SUFFIX"] or "",
    )

    # Write vars.bzl
    fci = "# Generated from llvm/CMakeLists.txt\n\n"
    fcd = "\nllvm_vars={\n"
    fct = "}\n"
    for k, v in c.items():
        if v != None:
            fci += '{} = "{}"\n'.format(k, v)
            fcd += '    "{}": "{}",\n'.format(k, v)
    repository_ctx.file("vars.bzl", content = fci + fcd + fct)

    # Write targets.bzl
    llvm_targets = repository_ctx.attr.targets
    repository_ctx.file(
        "llvm/targets.bzl",
        content = "llvm_targets = " + str(llvm_targets),
        executable = False,
    )

    # Write bolt targets
    bolt_targets = ["AArch64", "X86", "RISCV"]
    bolt_targets = [t for t in llvm_targets if t in bolt_targets]
    repository_ctx.file(
        "bolt/targets.bzl",
        content = "bolt_targets = " + str(bolt_targets),
        executable = False,
    )

llvm_configure = repository_rule(
    implementation = _llvm_configure_impl,
    local = True,
    configure = True,
    attrs = {
        "targets": attr.string_list(default = [
            "AArch64",
            "ARM",
            "RISCV",
            "X86",
            "NVPTX",
            "AMDGPU",
            "WebAssembly",
        ]),
    },
)

def _iree_extension_impl(module_ctx):
    """Implementation of the IREE module extension."""

    # Only configure repos once for the root module
    root_module = None
    for mod in module_ctx.modules:
        if mod.is_root:
            root_module = mod
            break

    if root_module == None:
        return

    # llvm-raw - points to the LLVM source tree
    new_local_repository(
        name = "llvm-raw",
        build_file_content = "# empty",
        path = "third_party/llvm-project",
    )

    # llvm-project - configured LLVM overlay
    llvm_configure(
        name = "llvm-project",
        targets = [
            "AArch64",
            "ARM",
            "RISCV",
            "X86",
            "NVPTX",
            "AMDGPU",
            "WebAssembly",
        ],
    )

    # Googletest
    local_repository(
        name = "com_google_googletest",
        path = "third_party/googletest",
    )

    # Flatcc
    new_local_repository(
        name = "com_github_dvidelabs_flatcc",
        build_file = "@iree_core//:build_tools/third_party/flatcc/BUILD.overlay",
        path = "third_party/flatcc",
    )

    # Vulkan headers
    new_local_repository(
        name = "vulkan_headers",
        build_file = "@iree_core//:build_tools/third_party/vulkan_headers/BUILD.overlay",
        path = "third_party/vulkan_headers",
    )

    # StableHLO
    local_repository(
        name = "stablehlo",
        path = "third_party/stablehlo",
    )

    # Benchmark
    local_repository(
        name = "com_google_benchmark",
        path = "third_party/benchmark",
    )

    # SPIRV-Cross
    new_local_repository(
        name = "spirv_cross",
        build_file = "@iree_core//:build_tools/third_party/spirv_cross/BUILD.overlay",
        path = "third_party/spirv_cross",
    )

    # Tracy
    new_local_repository(
        name = "tracy_client",
        build_file = "@iree_core//:build_tools/third_party/tracy_client/BUILD.overlay",
        path = "third_party/tracy",
    )

    # NCCL
    new_local_repository(
        name = "nccl",
        build_file = "@iree_core//:build_tools/third_party/nccl/BUILD.overlay",
        path = "third_party/nccl",
    )

    # HSA runtime headers
    new_local_repository(
        name = "hsa_runtime_headers",
        build_file = "@iree_core//:build_tools/third_party/hsa-runtime-headers/BUILD.overlay",
        path = "third_party/hsa-runtime-headers",
    )

    # WebGPU headers
    new_local_repository(
        name = "webgpu_headers",
        build_file = "@iree_core//:build_tools/third_party/webgpu-headers/BUILD.overlay",
        path = "third_party/webgpu-headers",
    )

    # CUDA auto-configuration
    cuda_auto_configure(
        name = "iree_cuda",
        iree_repo_alias = "@iree_core",
    )

iree_extension = module_extension(
    implementation = _iree_extension_impl,
)
