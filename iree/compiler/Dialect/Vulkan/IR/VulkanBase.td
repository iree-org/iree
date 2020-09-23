// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_DIALECT_VULKAN_BASE
#define IREE_DIALECT_VULKAN_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Vulkan dialect definition
//===----------------------------------------------------------------------===//

def VK_Dialect : Dialect {
  let name = "vk";
  let cppNamespace = "Vulkan";

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
class VK_IsKnownStrEnumCaseFor<string name> :
    CPred<"::mlir::iree_compiler::IREE::Vulkan::symbolize" # name # "("
          "$_self.cast<StringAttr>().getValue()).hasValue()">;

// Wrapper over base BitEnumAttr to set common fields.
class VK_BitEnumAttr<string name, string description,
                      list<BitEnumAttrCase> cases> :
    BitEnumAttr<name, description, cases> {
  let predicate = And<[I32Attr.predicate, VK_IsKnownBitEnumCaseFor<name>]>;
  let cppNamespace = "::mlir::iree_compiler::IREE::Vulkan";
}

// Wrapper over base I32EnumAttr to set common fields.
class VK_I32EnumAttr<string name, string description,
                      list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, description, cases> {
  let predicate = And<[I32Attr.predicate, VK_IsKnownIntEnumCaseFor<name>]>;
  let cppNamespace = "::mlir::iree_compiler::IREE::Vulkan";
}

// Wrapper over base StrEnumAttr to set common fields.
class VK_StrEnumAttr<string name, string description,
                      list<StrEnumAttrCase> cases> :
    StrEnumAttr<name, description, cases> {
  let predicate = And<[StrAttr.predicate, VK_IsKnownStrEnumCaseFor<name>]>;
  let cppNamespace = "::mlir::iree_compiler::IREE::Vulkan";
}

//===----------------------------------------------------------------------===//
// Target environment
//===----------------------------------------------------------------------===//

def VK_V_1_0 : I32EnumAttrCase<"V_1_0", 0, "v1.0">;
def VK_V_1_1 : I32EnumAttrCase<"V_1_1", 1, "v1.1">;
def VK_V_1_2 : I32EnumAttrCase<"V_1_2", 2, "v1.2">;

def VK_VersionAttr : VK_I32EnumAttr<"Version", "valid Vulkan version", [
    VK_V_1_0, VK_V_1_1, VK_V_1_2]>;

def VK_KHR_16bit_storage : StrEnumAttrCase<"VK_KHR_16bit_storage">;
def VK_KHR_8bit_storage : StrEnumAttrCase<"VK_KHR_8bit_storage">;
def VK_KHR_shader_float16_int8 : StrEnumAttrCase<"VK_KHR_shader_float16_int8">;
def VK_KHR_spirv_1_4 : StrEnumAttrCase<"VK_KHR_spirv_1_4">;
def VK_KHR_storage_buffer_storage_class : StrEnumAttrCase<"VK_KHR_storage_buffer_storage_class">;
def VK_KHR_variable_pointers: StrEnumAttrCase<"VK_KHR_variable_pointers">;

def VK_ExtensionAttr :
    VK_StrEnumAttr<"Extension", "supported Vulkan extension", [
      VK_KHR_16bit_storage, VK_KHR_8bit_storage, VK_KHR_shader_float16_int8,
      VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class,
      VK_KHR_variable_pointers
    ]>;

//===----------------------------------------------------------------------===//
// Subgroup features
//===----------------------------------------------------------------------===//

def VK_SF_Basic           : BitEnumAttrCase<"Basic", 0x001>;
def VK_SF_Vote            : BitEnumAttrCase<"Vote", 0x002>;
def VK_SF_Arithmetic      : BitEnumAttrCase<"Arithmetic", 0x004>;
def VK_SF_Ballot          : BitEnumAttrCase<"Ballot", 0x008>;
def VK_SF_Shuffle         : BitEnumAttrCase<"Shuffle", 0x010>;
def VK_SF_ShuffleRelative : BitEnumAttrCase<"ShuffleRelative", 0x020>;
def VK_SF_Clustered       : BitEnumAttrCase<"Clustered", 0x040>;
def VK_SF_Quad            : BitEnumAttrCase<"Quad", 0x080>;
def VK_SF_PartitionedNV   : BitEnumAttrCase<"PartitionedNV", 0x100>;

def VK_SubgroupFeatureAttr : VK_BitEnumAttr<
  "SubgroupFeature", "supported Vulkan subgroup feature", [
    VK_SF_Basic, VK_SF_Vote, VK_SF_Arithmetic, VK_SF_Ballot, VK_SF_Shuffle,
    VK_SF_ShuffleRelative, VK_SF_Clustered, VK_SF_Quad, VK_SF_PartitionedNV
  ]>;

#endif  // IREE_DIALECT_VULKAN_BASE
