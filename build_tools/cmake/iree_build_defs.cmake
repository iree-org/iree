# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Bazel's variable IREE_ENABLE_TRACING set in iree/base/build_defs.oss.bzl
if(${IREE_ENABLE_TRACING})
  set(IREE_BASE_TRACING_DEPS
    iree::base::tracing_enabled
    # TODO: Add dependency
    # "@com_google_tracing_framework_cpp//:wtf_enable"
  )
else()
  set(IREE_BASE_TRACING_DEPS
    iree::base::tracing_disabled
  )
endif()

# Bazel's variable TARGET_COMPILER_BACKENDS set in iree/tools/build_defs.oss.bzl
set(TARGET_COMPILER_BACKENDS
  iree::compiler::Dialect::HAL::Target::LegacyInterpreter
  iree::compiler::Dialect::HAL::Target::VulkanSPIRV
)

# Bazel's variable PLATFORM_VULKAN_TEST_DEPS set in iree/build_defs.oss.bzl
set(PLATFORM_VULKAN_TEST_DEPS
  iree::testing::gtest_main
)