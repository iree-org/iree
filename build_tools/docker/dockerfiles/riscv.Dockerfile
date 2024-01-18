# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE towards RISCV using CMake.

FROM gcr.io/iree-oss/base@sha256:61e9aae211007dbad95e1f429e9e5121fd5968c204791038424979c21146cf75 AS install-riscv
WORKDIR /install-riscv
RUN wget --no-verbose "https://storage.googleapis.com/iree-shared-files/toolchain_iree_manylinux_2_28_20231012.tar.gz"
RUN tar -xf "toolchain_iree_manylinux_2_28_20231012.tar.gz" -C /usr/src/
RUN wget --no-verbose "https://storage.googleapis.com/iree-shared-files/toolchain_iree_rv32imf_manylinux_2_28_20231012.tar.gz"
RUN tar -xf "toolchain_iree_rv32imf_manylinux_2_28_20231012.tar.gz" -C /usr/src/
RUN wget --no-verbose "https://storage.googleapis.com/iree-shared-files/qemu-riscv_8.1.2_manylinux_2.28_20231026.tar.gz"
RUN tar -xf "qemu-riscv_8.1.2_manylinux_2.28_20231026.tar.gz" -C /usr/src/
# Old qemu-v5.2.0 to support embedded elf (without memory protection)
RUN mkdir -p /usr/src/qemu-v5.2.0
RUN wget --no-verbose "https://storage.googleapis.com/iree-shared-files/qemu-riscv.tar.gz"
RUN tar -xf "qemu-riscv.tar.gz" -C /usr/src/qemu-v5.2.0

FROM gcr.io/iree-oss/base@sha256:61e9aae211007dbad95e1f429e9e5121fd5968c204791038424979c21146cf75 AS final
COPY --from=install-riscv "/usr/src/toolchain_iree" "/usr/src/toolchain_iree"
COPY --from=install-riscv "/usr/src/toolchain_iree_rv32imf" "/usr/src/toolchain_iree_rv32imf"
COPY --from=install-riscv "/usr/src/qemu-riscv" "/usr/src/qemu-riscv"
COPY --from=install-riscv "/usr/src/qemu-v5.2.0/qemu-riscv/qemu-riscv32" "/usr/src/qemu-riscv/qemu-riscv32-v5.2.0"
ENV RISCV_RV64_LINUX_TOOLCHAIN_ROOT="/usr/src/toolchain_iree"
ENV RISCV_RV32_NEWLIB_TOOLCHAIN_ROOT="/usr/src/toolchain_iree_rv32imf"
ENV QEMU_RV64_BIN="/usr/src/qemu-riscv/qemu-riscv64"
ENV QEMU_RV32_BIN="/usr/src/qemu-riscv/qemu-riscv32"
ENV QEMU_RV32_V5_2_BIN="/usr/src/qemu-riscv/qemu-riscv32-v5.2.0"
