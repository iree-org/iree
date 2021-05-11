# Copyright 2021 Nod Labs
# Copyright 2019 Google LLC
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

iree_cc_binary(
  NAME
    iree-opt-rocm
  SRCS
    "iree-opt-rocm-main.cc"
  DEPS
    iree::tools::iree_opt_main
    experimental::rocm::Conversion::init_rocm_conversions
    experimental::rocm::Conversion::LinalgToROCDL::LinalgToROCDL
    experimental::rocm::Dialect::HAL::Target::ROCM::ROCM
    LLVMSupport
    MLIRIR
    MLIROptLib
    MLIRSupport
  PUBLIC
)
