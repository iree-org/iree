// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VULKAN_BASE
#define IREE_DIALECT_VULKAN_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Vulkan dialect definition
//===----------------------------------------------------------------------===//

def VK_Dialect : Dialect {
  let name = "vk";
  let cppNamespace = "::mlir::iree_compiler::IREE::Vulkan";

  let summary = "The Vulkan dialect in IREE";
  let description = [{
    Vulkan is a new generation graphics and compute API that provides
    high-efficiency, cross-platform access to modern GPUs used in a wide
    variety of devices from PCs and consoles to mobile phones and embedded
    platforms. See https://www.khronos.org/vulkan for more details regarding
    Vulkan itself.

    This is not a full-fledged Vulkan dialect that models common Vulkan concepts
    in intermediate representation to be amenable to compiler analysis and
    transformation. IREE has the HAL dialect for that purpose. Instead, this
    dialect contains useful utilities for targeting Vulkan both in CodeGen and
    runtime.
  }];
}

//===----------------------------------------------------------------------===//
// Utility definitions
//===----------------------------------------------------------------------===//

// A predicate that checks whether `$_self` is a known enum case for the
// enum class with `name`.
class VK_IsKnownBitEnumCaseFor<string name> :
    CPred<"::mlir::iree_compiler::IREE::Vulkan::symbolize" # name # "("
          "$_self.cast<IntegerAttr>().getValue().getZExtValue()).hasValue()">;
class VK_IsKnownIntEnumCaseFor<string name> :
    CPred<"::mlir::iree_compiler::IREE::Vulkan::symbolize" # name # "("
          "$_self.cast<IntegerAttr>().getValue().getZExtValue()).hasValue()">;

// Wrapper over base I32BitEnumAttr to set common fields.
class VK_BitEnumAttr<string name, string description,
                     list<I32BitEnumAttrCase> cases> :
    I32BitEnumAttr<name, description, cases> {
  let predicate = And<[I32Attr.predicate, VK_IsKnownBitEnumCaseFor<name>]>;
  let cppNamespace = "::mlir::iree_compiler::IREE::Vulkan";
}

class VK_I32Enum<string name, string description, list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, description, cases> {
  let predicate = And<[I32Attr.predicate, VK_IsKnownIntEnumCaseFor<name>]>;
  let cppNamespace = "::mlir::iree_compiler::IREE::Vulkan";
}

class VK_I32EnumAttr<string name, string description, string mnemonic,
                     list<I32EnumAttrCase> cases> :
    EnumAttr<VK_Dialect, I32EnumAttr<name, description, cases>, mnemonic> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Vulkan";
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Target environment
//===----------------------------------------------------------------------===//

def VK_V_1_0 : I32EnumAttrCase<"V_1_0", 0, "v1.0">;
def VK_V_1_1 : I32EnumAttrCase<"V_1_1", 1, "v1.1">;
def VK_V_1_2 : I32EnumAttrCase<"V_1_2", 2, "v1.2">;
def VK_V_1_3 : I32EnumAttrCase<"V_1_3", 3, "v1.3">;

def VK_VersionAttr : VK_I32Enum<"Version", "valid Vulkan version", [
    VK_V_1_0, VK_V_1_1, VK_V_1_2, VK_V_1_3
]>;

def VK_KHR_16bit_storage : I32EnumAttrCase<"VK_KHR_16bit_storage", 0>;
def VK_KHR_8bit_storage : I32EnumAttrCase<"VK_KHR_8bit_storage", 1>;
def VK_KHR_shader_float16_int8 : I32EnumAttrCase<"VK_KHR_shader_float16_int8", 2>;
def VK_KHR_shader_integer_dot_product : I32EnumAttrCase<"VK_KHR_shader_integer_dot_product", 3>;
def VK_KHR_spirv_1_4 : I32EnumAttrCase<"VK_KHR_spirv_1_4", 4>;
def VK_KHR_storage_buffer_storage_class : I32EnumAttrCase<"VK_KHR_storage_buffer_storage_class", 5>;
def VK_KHR_variable_pointers: I32EnumAttrCase<"VK_KHR_variable_pointers", 6>;
def VK_EXT_subgroup_size_control : I32EnumAttrCase<"VK_EXT_subgroup_size_control", 7>;
def VK_NV_cooperative_matrix : I32EnumAttrCase<"VK_NV_cooperative_matrix", 8>;

def VK_ExtensionAttr :
    VK_I32EnumAttr<"Extension", "supported Vulkan extension", "extension", [
      VK_KHR_16bit_storage, VK_KHR_8bit_storage, VK_KHR_shader_float16_int8,
      VK_KHR_shader_integer_dot_product, VK_KHR_spirv_1_4,
      VK_KHR_storage_buffer_storage_class, VK_KHR_variable_pointers,
      VK_EXT_subgroup_size_control, VK_NV_cooperative_matrix
    ]>;

//===----------------------------------------------------------------------===//
// Target triple
//===----------------------------------------------------------------------===//

def VK_TTA_Unknown   : I32EnumAttrCase<"Unknown", 0, "unknown">;
// Software emulated GPU
def VK_TTA_CPU       : I32EnumAttrCase<"CPU", 1, "cpu">;
// AMD GPU
def VK_TTA_RDNAv1    : I32EnumAttrCase<"AMD_RDNAv1", 100, "rdna1">;
def VK_TTA_RDNAv2    : I32EnumAttrCase<"AMD_RDNAv2", 101, "rdna2">;
def VK_TTA_RDNAv3    : I32EnumAttrCase<"AMD_RDNAv3", 102, "rdna3">;
// Apple Silicon GPU
def VK_TTA_M1        : I32EnumAttrCase<"Apple_M1", 200, "m1">;
// ARM Mali GPU
def VK_TTA_Valhall   : I32EnumAttrCase<"ARM_Valhall", 300, "valhall">;
// NVIDIA GPU
def VK_TTA_Turing    : I32EnumAttrCase<"NV_Turing", 400, "turing">;
def VK_TTA_Ampere    : I32EnumAttrCase<"NV_Ampere", 401, "ampere">;
def VK_TTA_Pascal    : I32EnumAttrCase<"NV_Pascal", 402, "pascal">;
// Qualcomm Adreno GPU
def VK_TTA_Adreno    : I32EnumAttrCase<"QC_Adreno", 500, "adreno">;
// Intel ARC GPU
def VK_TTA_Arc       : I32EnumAttrCase<"Intel_Arc", 600, "arc">;

def VK_TargetArchAttr : VK_I32Enum<
  "TargetTripleArch", "recognized target architecture", [
    VK_TTA_Unknown, VK_TTA_CPU, VK_TTA_RDNAv1, VK_TTA_RDNAv2,
    VK_TTA_RDNAv3, VK_TTA_M1, VK_TTA_Valhall, VK_TTA_Turing, VK_TTA_Ampere,
    VK_TTA_Pascal, VK_TTA_Adreno, VK_TTA_Arc,
  ]>;

def VK_TTP_Unknown     : I32EnumAttrCase<"Unknown", 0, "unknown">;
// Qualcomm Adreno GPU
def VK_TTP_Adreno640   : I32EnumAttrCase<"Adreno_640", 100, "a640">;
def VK_TTP_Adreno650   : I32EnumAttrCase<"Adreno_650", 101, "a650">;
def VK_TTP_Adreno660   : I32EnumAttrCase<"Adreno_660", 102, "a660">;
// Software emulated GPU
def VK_TTP_SwiftShader : I32EnumAttrCase<"SwiftShader", 200, "swiftshader">;
// Translation layers
def VK_TTP_MoltenVK    : I32EnumAttrCase<"MoltenVK", 300, "moltenvk">;

def VK_TargetProductAttr : VK_I32Enum<
  "TargetTripleProduct", "recognized target product", [
    VK_TTP_Unknown, VK_TTP_Adreno650, VK_TTP_Adreno660, VK_TTP_SwiftShader,
    VK_TTP_MoltenVK,
  ]>;

def VK_TTOS_Unknown   : I32EnumAttrCase<"Unknown", 0, "unknown">;
def VK_TTOS_Linux     : I32EnumAttrCase<"Linux", 1, "linux">;
def VK_TTOS_iOS       : I32EnumAttrCase<"iOS", 2, "iOS">;
def VK_TTOS_macOS     : I32EnumAttrCase<"macOS", 3, "macos">;
def VK_TTOS_Windows   : I32EnumAttrCase<"Windows", 4, "windows">;
// API Level 30 => Android 11
def VK_TTOS_Android30 : I32EnumAttrCase<"Android30", 5, "android30">;
// API Level 31 => Android 12
def VK_TTOS_Android31 : I32EnumAttrCase<"Android31", 6, "android31">;

def VK_TargetOSAttr : VK_I32Enum<
  "TargetTripleOS", "recognized target operating system", [
    VK_TTOS_Unknown, VK_TTOS_Linux, VK_TTOS_iOS, VK_TTOS_macOS,
    VK_TTOS_Windows, VK_TTOS_Android30, VK_TTOS_Android31,
  ]>;

//===----------------------------------------------------------------------===//
// Subgroup features
//===----------------------------------------------------------------------===//

def VK_SF_Basic           : I32BitEnumAttrCase<"Basic", 0x001>;
def VK_SF_Vote            : I32BitEnumAttrCase<"Vote", 0x002>;
def VK_SF_Arithmetic      : I32BitEnumAttrCase<"Arithmetic", 0x004>;
def VK_SF_Ballot          : I32BitEnumAttrCase<"Ballot", 0x008>;
def VK_SF_Shuffle         : I32BitEnumAttrCase<"Shuffle", 0x010>;
def VK_SF_ShuffleRelative : I32BitEnumAttrCase<"ShuffleRelative", 0x020>;
def VK_SF_Clustered       : I32BitEnumAttrCase<"Clustered", 0x040>;
def VK_SF_Quad            : I32BitEnumAttrCase<"Quad", 0x080>;
def VK_SF_PartitionedNV   : I32BitEnumAttrCase<"PartitionedNV", 0x100>;

def VK_SubgroupFeatureAttr : VK_BitEnumAttr<
  "SubgroupFeature", "supported Vulkan subgroup feature", [
    VK_SF_Basic, VK_SF_Vote, VK_SF_Arithmetic, VK_SF_Ballot, VK_SF_Shuffle,
    VK_SF_ShuffleRelative, VK_SF_Clustered, VK_SF_Quad, VK_SF_PartitionedNV
  ]>;

def VK_SNV_Device      : I32EnumAttrCase<"Device", 1>;
def VK_SNV_Workgroup   : I32EnumAttrCase<"Workgroup", 2>;
def VK_SNV_Subgroup    : I32EnumAttrCase<"Subgroup", 3>;
def VK_SNV_QueueFamily : I32EnumAttrCase<"QueueFamily", 5>;

def VK_ScopeNV_Attr :
  VK_I32EnumAttr<"ScopeNV", "valid VkScopeNV", "scope", [
    VK_SNV_Device, VK_SNV_Workgroup, VK_SNV_Subgroup,
    VK_SNV_QueueFamily
  ]>;

#endif  // IREE_DIALECT_VULKAN_BASE
