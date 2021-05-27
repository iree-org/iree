#!/bin/bash
# Copyright 2021 Google LLC
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

set -x
set -e

ROOT_DIR=$(git rev-parse --show-toplevel)

# TODO(benvanik): document usage.
# Basically just copy/paste/rename by hand for now.

CMD=(
  iree/tools/iree-translate
    -iree-mlir-to-vm-bytecode-module
    ${ROOT_DIR}/iree/samples/simple_embedding/simple_embedding_test.mlir
    -o=${ROOT_DIR}/../iree-tmp/simple_embedding_test_dylib_llvm_aot.vmfb

    -iree-hal-target-backends=dylib-llvm-aot
    -iree-llvm-link-embedded=true
    -iree-llvm-debug-symbols=false

    -iree-llvm-target-triple=x86_64-pc-linux-elf

    #-iree-llvm-target-triple=aarch64-pc-linux-elf

    #-iree-llvm-target-triple=riscv32-pc-linux-elf
    #-iree-llvm-target-cpu=generic-rv32
    #-iree-llvm-target-cpu-features=+f
    #-iree-llvm-target-abi=ilp32f
    #-iree-llvm-target-float-abi=hard

    #-iree-llvm-target-triple=riscv64-pc-linux-elf
    #-iree-llvm-target-cpu=generic-rv64
    #-iree-llvm-target-cpu=sifive-u74
    #-iree-llvm-target-abi=lp64d

    #-iree-llvm-keep-linker-artifacts
)
"${CMD[@]}"
