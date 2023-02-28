# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE towards RISCV using CMake.

FROM gcr.io/iree-oss/base@sha256:f26a5aa5f8d3705c6b80c71d04fafb360861f1907bdd1b1f5f19480b6192664e AS install-riscv
WORKDIR /install-riscv
RUN wget --no-verbose "https://storage.googleapis.com/iree-shared-files/toolchain_iree_20220918.tar.gz"
RUN tar -xf "toolchain_iree_20220918.tar.gz" -C /usr/src/
RUN wget --no-verbose "https://storage.googleapis.com/iree-shared-files/toolchain_iree_rv32_20220918.tar.gz"
RUN tar -xf "toolchain_iree_rv32_20220918.tar.gz" -C /usr/src/
RUN wget --no-verbose "https://storage.googleapis.com/iree-shared-files/qemu-riscv.tar.gz"
RUN tar -xf "qemu-riscv.tar.gz" -C /usr/src/

FROM gcr.io/iree-oss/base@sha256:f26a5aa5f8d3705c6b80c71d04fafb360861f1907bdd1b1f5f19480b6192664e AS final
COPY --from=install-riscv "/usr/src/toolchain_iree" "/usr/src/toolchain_iree"
COPY --from=install-riscv "/usr/src/toolchain_iree_rv32imf" "/usr/src/toolchain_iree_rv32imf"
COPY --from=install-riscv "/usr/src/qemu-riscv" "/usr/src/qemu-riscv"
ENV RISCV_RV64_LINUX_TOOLCHAIN_ROOT="/usr/src/toolchain_iree"
ENV RISCV_RV32_NEWLIB_TOOLCHAIN_ROOT="/usr/src/toolchain_iree_rv32imf"
ENV QEMU_RV64_BIN="/usr/src/qemu-riscv/qemu-riscv64"
ENV QEMU_RV32_BIN="/usr/src/qemu-riscv/qemu-riscv32"
