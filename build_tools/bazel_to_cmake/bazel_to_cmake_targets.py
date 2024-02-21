# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List
import re


class TargetConverter:
    def __init__(self, repo_map: Dict[str, str]):
        self._explicit_target_mapping = {}
        self._repo_map = repo_map

        iree_core_repo = self._repo_alias("@iree_core")
        self._update_target_mappings(
            {
                # Internal utilities to emulate various binary/library options.
                f"{iree_core_repo}//build_tools:pthreads": [],
                f"{iree_core_repo}//build_tools:dl": ["${CMAKE_DL_LIBS}"],
                f"{iree_core_repo}//compiler/src/iree/compiler/API:CAPI": [
                    "IREECompilerCAPILib"
                ],
                # IREE llvm-external-projects
                f"{iree_core_repo}//llvm-external-projects/iree-dialects:CAPI": [
                    "IREEDialectsCAPI"
                ],
                # Disable all hard-coded codegen targets (they are expanded dynamically
                # in CMake).
                "@llvm-project//llvm:AArch64AsmParser": ["IREELLVMCPUTargetDeps"],
                "@llvm-project//llvm:AArch64CodeGen": ["IREELLVMCPUTargetDeps"],
                "@llvm-project//llvm:ARMAsmParser": ["IREELLVMCPUTargetDeps"],
                "@llvm-project//llvm:ARMCodeGen": ["IREELLVMCPUTargetDeps"],
                "@llvm-project//llvm:RISCVAsmParser": ["IREELLVMCPUTargetDeps"],
                "@llvm-project//llvm:RISCVCodeGen": ["IREELLVMCPUTargetDeps"],
                "@llvm-project//llvm:WebAssemblyAsmParser": ["IREELLVMCPUTargetDeps"],
                "@llvm-project//llvm:WebAssemblyCodeGen": ["IREELLVMCPUTargetDeps"],
                "@llvm-project//llvm:X86AsmParser": ["IREELLVMCPUTargetDeps"],
                "@llvm-project//llvm:X86CodeGen": ["IREELLVMCPUTargetDeps"],
                # Clang
                "@llvm-project//clang": ["${IREE_CLANG_TARGET}"],
                # LLD
                "@llvm-project//lld": ["${IREE_LLD_TARGET}"],
                "@llvm-project//lld:COFF": ["lldCOFF"],
                "@llvm-project//lld:Common": ["lldCommon"],
                "@llvm-project//lld:ELF": ["lldELF"],
                "@llvm-project//lld:MachO": ["lldMachO"],
                "@llvm-project//lld:Wasm": ["lldWasm"],
                # LLVM
                "@llvm-project//llvm:config": [],
                "@llvm-project//llvm:IPO": ["LLVMipo"],
                "@llvm-project//llvm:FileCheck": ["FileCheck"],
                "@llvm-project//llvm:not": ["not"],
                "@llvm-project//llvm:llvm-link": ["${IREE_LLVM_LINK_TARGET}"],
                "@llvm-project//llvm:NVPTXUtilsAndDesc": [
                    "LLVMNVPTXDesc",
                ],
                # MLIR
                "@llvm-project//mlir:AllPassesAndDialects": ["MLIRAllDialects"],
                "@llvm-project//mlir:CommonFolders": [""],
                "@llvm-project//mlir:DialectUtils": [""],
                "@llvm-project//mlir:GPUDialect": ["MLIRGPUDialect"],
                "@llvm-project//mlir:GPUTransforms": ["MLIRGPUTransforms"],
                "@llvm-project//mlir:LinalgOpsIncGen": ["MLIRLinalgOpsIncGenLib"],
                "@llvm-project//mlir:LinalgStructuredOpsIncGen": [
                    "MLIRLinalgStructuredOpsIncGenLib"
                ],
                "@llvm-project//mlir:ShapeTransforms": ["MLIRShapeOpsTransforms"],
                "@llvm-project//mlir:ToLLVMIRTranslation": ["MLIRTargetLLVMIRExport"],
                "@llvm-project//mlir:mlir-translate": ["mlir-translate"],
                "@llvm-project//mlir:MlirLspServerLib": ["MLIRLspServerLib"],
                "@llvm-project//mlir:MlirTableGenMain": ["MLIRTableGen"],
                "@llvm-project//mlir:MlirOptLib": ["MLIROptLib"],
                "@llvm-project//mlir:VectorOps": ["MLIRVector"],
                # StableHLO.
                "@stablehlo//:chlo_ops": [
                    "ChloOps",
                ],
                "@stablehlo//:stablehlo_ops": [
                    "StablehloOps",
                ],
                "@stablehlo//:broadcast_utils": [
                    "StablehloBroadcastUtils",
                ],
                # NCCL
                "@nccl//:headers": [
                    "nccl::headers",
                ],
                # Tracy.
                "@tracy_client//:runtime": ["tracy_client::runtime"],
                # Vulkan
                "@vulkan_headers": ["Vulkan::Headers"],
                # Misc single targets
                "@com_google_benchmark//:benchmark": ["benchmark"],
                "@com_github_dvidelabs_flatcc//:flatcc": ["flatcc"],
                "@com_github_dvidelabs_flatcc//:parsing": ["flatcc::parsing"],
                "@com_github_dvidelabs_flatcc//:runtime": ["flatcc::runtime"],
                "@com_google_googletest//:gtest": ["gmock", "gtest"],
                "@spirv_cross//:spirv_cross_lib": ["spirv-cross-msl"],
                "@cpuinfo": ["${IREE_CPUINFO_TARGET}"],
                "@webgpu_headers": [],
            }
        )

        self._initialize()

    def _initialize(self):
        pass

    def _repo_alias(self, repo_name: str) -> str:
        """Returns the prefix of a repo (i.e. '@iree_core') given the repo map."""
        return self._repo_map.get(repo_name, repo_name)

    def _update_target_mappings(self, mappings: Dict[str, List[str]]):
        self._explicit_target_mapping.update(mappings)

    def _convert_mlir_target(self, target):
        # Default to a pattern substitution approach.
        # Take "MLIR" and append the name part of the full target identifier, e.g.
        #   "@llvm-project//mlir:IR"   -> "MLIRIR"
        #   "@llvm-project//mlir:Pass" -> "MLIRPass"
        # MLIR does not have header-only targets apart from the libraries. Here
        # we redirect any request for a CAPI{Name}Headers to a target within IREE
        # that sets this up.
        label = target.rsplit(":")[-1]
        if label.startswith("CAPI") and label.endswith("Headers"):
            return [f"IREELLVMIncludeSetup"]
        else:
            return [f"MLIR{label}"]

    def _convert_llvm_target(self, target):
        # Default to a pattern substitution approach.
        # Prepend "LLVM" to the Bazel target name.
        #   "@llvm-project//llvm:AsmParser" -> "LLVMAsmParser"
        #   "@llvm-project//llvm:Core" -> "LLVMCore"
        return ["LLVM" + target.rsplit(":")[-1]]

    def _convert_iree_cuda_target(self, target):
        # Convert like:
        #   @iree_cuda//:libdevice_embedded -> iree_cuda::libdevice_embedded
        label = target.rsplit(":")[-1]
        return [f"iree_cuda::{label}"]

    def _convert_iree_dialects_target(self, target):
        # Just take the target name as-is.
        return [target.rsplit(":")[-1]]

    def _convert_to_cmake_path(self, bazel_path_fragment: str) -> str:
        cmake_path = bazel_path_fragment
        # Bazel `//iree/base`     -> CMake `iree::base`
        # Bazel `//iree/base:foo` -> CMake `iree::base::foo`
        if cmake_path.startswith("//"):
            cmake_path = cmake_path[len("//") :]
        cmake_path = cmake_path.replace(":", "::")  # iree/base::foo or ::foo
        cmake_path = cmake_path.replace("/", "::")  # iree::base
        return cmake_path

    def convert_target(self, target):
        """Converts a Bazel target to a list of CMake targets.

        IREE targets are expected to follow a standard form between Bazel and CMake
        that facilitates conversion. External targets *may* have their own patterns,
        or they may be purely special cases.

        Multiple target in Bazel may map to a single target in CMake and a Bazel
        target may map to multiple CMake targets.

        Returns:
          A list of converted targets if it was successfully converted.

        Raises:
          KeyError: No conversion was found for the target.
        """
        iree_core_repo = self._repo_alias("@iree_core")
        if target in self._explicit_target_mapping:
            return self._explicit_target_mapping[target]
        if target.startswith("@llvm-project//llvm"):
            return self._convert_llvm_target(target)
        if target.startswith("@llvm-project//mlir"):
            return self._convert_mlir_target(target)
        if target.startswith("@iree_cuda//"):
            return self._convert_iree_cuda_target(target)
        if target.startswith(f"{iree_core_repo}//"):
            return self._convert_iree_core_target(target)
        if target.startswith("@"):
            raise KeyError(f"No conversion found for target '{target}'")

        # Pass through package-relative targets
        #   :target_name
        #   file_name.txt
        if target.startswith(":") or (":" not in target and not target.startswith("/")):
            return [self._convert_to_cmake_path(target)]

        return self._convert_unmatched_target(target)

    def _convert_iree_core_target(self, target):
        iree_core_repo = self._repo_alias("@iree_core")
        if target.startswith(f"{iree_core_repo}//llvm-external-projects/iree-dialects"):
            return self._convert_iree_dialects_target(target)

        # IREE root paths map to package names based on explicit rules.
        #   * src/iree/ directories (compiler/src/iree/ and runtime/src/iree/)
        #     creating their own root paths by trimming down to just "iree"
        #   * tools/ uses an empty root, for binary targets names like "iree-compile"
        #   * other top level directories add back an 'iree' prefix
        # If changing these, make the corresponding change in iree_macros.cmake
        # (iree_package_ns function).

        # Map //compiler/src/iree/(.*) -> iree::\1 (i.e. iree::compiler::\1)
        m = re.match(f"^{iree_core_repo}//compiler/src/iree/(.+)", target)
        if m:
            return ["iree::" + self._convert_to_cmake_path(m.group(1))]

        # Map //runtime/src/iree/(.*) -> iree::\1
        m = re.match(f"^{iree_core_repo}//runtime/src/iree/(.+)", target)
        if m:
            return ["iree::" + self._convert_to_cmake_path(m.group(1))]

        # Map //tools/(.*) -> \1
        m = re.match(f"^{iree_core_repo}//tools[/|:](.+)", target)
        if m:
            return [self._convert_to_cmake_path(m.group(1))]

        return self._convert_unmatched_target(target)

    def _convert_unmatched_target(self, target: str) -> str:
        """Converts unmatched targets in a repo specific way."""
        raise ValueError(f"No target matching for {target}")
