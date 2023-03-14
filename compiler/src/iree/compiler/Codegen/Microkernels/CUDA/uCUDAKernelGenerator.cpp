// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "uCUDAContract.h"

// This program is responsible of generating microkernel generator

void generateUndefs(std::ofstream& ofs) {
  ofs << "#undef ELEMENT_A\n";
  ofs << "#undef ELEMENT_B\n";
  ofs << "#undef ELEMENT_C\n";
  ofs << "#undef TILE_M\n";
  ofs << "#undef TILE_N\n";
  ofs << "#undef TILE_K\n";
  ofs << "#undef WARP_M\n";
  ofs << "#undef WARP_N\n";
  ofs << "#undef INST_M\n";
  ofs << "#undef INST_N\n";
  ofs << "#undef INST_K\n";
  ofs << "#undef STAGES\n";
  ofs << "#undef HAS_LINALG_FILL\n";
  ofs << "#undef WRITEBACK_TO_GLOBAL\n";
}

void generateKernel(std::ofstream& ofs, uGPUKernel ukernel) {
  ofs << "\n";
  ofs << "//===-------------------------------------------------------===//\n";
  ofs << "\n";
  generateUndefs(ofs);
  ofs << "#define ELEMENT_A  " << ukernel.LHS << "\n";
  ofs << "#define ELEMENT_B  " << ukernel.RHS << "\n";
  ofs << "#define ELEMENT_C  " << ukernel.RESULT << "\n";
  ofs << "#define TILE_M  " << ukernel.ThreadblockShape[0] << "\n";
  ofs << "#define TILE_N  " << ukernel.ThreadblockShape[1] << "\n";
  ofs << "#define TILE_K  " << ukernel.ThreadblockShape[2] << "\n";
  ofs << "#define WARP_M  " << ukernel.WarpShape[0] << "\n";
  ofs << "#define WARP_N  " << ukernel.WarpShape[1] << "\n";
  ofs << "#define INST_M  " << ukernel.InstShape[0] << "\n";
  ofs << "#define INST_N  " << ukernel.InstShape[1] << "\n";
  ofs << "#define INST_K  " << ukernel.InstShape[2] << "\n";
  ofs << "#define STAGES  " << ukernel.stages << "\n";
  ofs << "#define HAS_LINALG_FILL  "
      << (ukernel.has_linalg_fill ? "true" : "false") << "\n";
  ofs << "#define WRITEBACK_TO_GLOBAL  "
      << (ukernel.writeback_to_global ? "true" : "false") << "\n";

  ofs << "#include GPUK_MATMUL_HEADER\n";
}

void initFile(std::ofstream& ofs) {
  ofs << "//===-------------------------------------------------------===//\n";
  ofs << "// Auto generated file - Explicit instantiations\n";
  ofs << "//===-------------------------------------------------------===//\n";
  ofs << "\n";
  ofs << "#define GPUK_MATMUL_HEADER \"uCUDAGemmTemplate.cuh\"\n";
  ofs << "#define KERNEL_NAME __iree_ucuda_linalg_matmul\n";
}

int main(int argc, char* argv[]) {
  std::ofstream ofs;
  std::string fname;
  if (argc != 2) {
    std::cout << "Define a file name" << std::endl;
  }
  fname = argv[1];
  ofs.open(fname, std::ios::out | std::ios::app | std::ios::ate);

  if (!ofs) {
    std::cout << "Unable to open file to write" << std::endl;
    return -1;
  }
  initFile(ofs);

  for (uGPUKernel ukernel : AllContracts.ukernels) {
    generateKernel(ofs, ukernel);
  }
  ofs.close();
  return 0;
}