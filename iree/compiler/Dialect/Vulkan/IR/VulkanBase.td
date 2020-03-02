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

    The Vulkan dialect defines common Vulkan concepts in intermediate
    representation to be amenable to compiler analysis and transformation.
  }];
}

//===----------------------------------------------------------------------===//
// Utility definitions
//===----------------------------------------------------------------------===//

// A predicate that checks whether `$_self` is a known enum case for the
// enum class with `name`.
class VK_IsKnownIntEnumCaseFor<string name> :
    CPred<"::mlir::iree_compiler::IREE::vulkan::symbolize" # name # "("
          "$_self.cast<IntegerAttr>().getValue().getZExtValue()).hasValue()">;
class VK_IsKnownStrEnumCaseFor<string name> :
    CPred<"::mlir::iree_compiler::IREE::vulkan::symbolize" # name # "("
          "$_self.cast<StringAttr>().getValue()).hasValue()">;

// Wrapper over base I32EnumAttr to set common fields.
class VK_I32EnumAttr<string name, string description,
                      list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, description, cases> {
  let predicate = And<[
    IntegerAttrBase<I32, "">.predicate,
    VK_IsKnownIntEnumCaseFor<name>,
  ]>;
  let cppNamespace = "::mlir::iree_compiler::IREE::Vulkan";
}

// Wrapper over base StrEnumAttr to set common fields.
class VK_StrEnumAttr<string name, string description,
                      list<StrEnumAttrCase> cases> :
    StrEnumAttr<name, description, cases> {
  let predicate = And<[
    StrAttr.predicate,
    VK_IsKnownStrEnumCaseFor<name>,
  ]>;
  let cppNamespace = "::mlir::iree_compiler::IREE::Vulkan";
}

//===----------------------------------------------------------------------===//
// Target Environment
//===----------------------------------------------------------------------===//

def VK_V_1_0 : I32EnumAttrCase<"V_1_0", 0, "v1.0">;
def VK_V_1_1 : I32EnumAttrCase<"V_1_1", 1, "v1.1">;
def VK_V_1_2 : I32EnumAttrCase<"V_1_2", 2, "v1.2">;

def VK_VersionAttr : VK_I32EnumAttr<"Version", "valid Vulkan version", [
    VK_V_1_0, VK_V_1_1, VK_V_1_2]>;

def VK_KHR_spirv_1_4 : StrEnumAttrCase<"VK_KHR_spirv_1_4">;
def VK_KHR_storage_buffer_storage_class : StrEnumAttrCase<
    "VK_KHR_storage_buffer_storage_class">;

def VK_ExtensionnAttr :
    VK_StrEnumAttr<"Extension", "supported Vulkan extension", [
      VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class
    ]>;

#endif  // IREE_DIALECT_VULKAN_BASE
