// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UCUDA_GEMM_CUTLASS
#define UCUDA_GEMM_CUTLASS

#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

#ifdef DEBUG_CUTLASS
#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"
#endif

template <class ElementA, class ElementB, class ElementC, int Tile_m,
          int Tile_n, int Tile_k, int Warp_m, int Warp_n, int Inst_m,
          int Inst_n, int Inst_k, bool hasLinalgFill>
__forceinline__ __device__ void gemm_ukernel(
    ElementA* lhs, int64_t lhs_offset, int64_t lhs_dim2, ElementB* rhs,
    int64_t rhs_offset, int64_t rhs_dim2, ElementC* res, int64_t res_offset,
    int64_t res_dim2, ElementC* shmem, ElementC initValue) {
  ElementA* plhs = (ElementA*)__builtin_assume_aligned(lhs, 256);
  ElementB* prhs = (ElementB*)__builtin_assume_aligned(rhs, 256);
  ElementC* pres = (ElementC*)__builtin_assume_aligned(res, 256);
  using ElementAccumulator = ElementC;
  // todo(guray) Can be templatized
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ThreadblockShape = cutlass::gemm::GemmShape<Tile_m, Tile_n, Tile_k>;
  using WarpShape = cutlass::gemm::GemmShape<Warp_m, Warp_n, Tile_k>;
  using InstructionShape = cutlass::gemm::GemmShape<Inst_m, Inst_n, Inst_k>;

  // todo(guray) maybe templatize kStages?
  int const kStages = 3;
  int const kAlignmentA = 4;
  int const kAlignmentB = 4;

  // CUTLASS Threadblock-level matmul operator and global memory tile
  // iterators
  using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape,
      kStages, cutlass::arch::OpMultiplyAdd>;

  // CUTLASS Threadblock-level multistage matrix multiply-accumulate
  // pipeline
  using ThreadblockMma = typename DefaultMma::ThreadblockMma;
  using IteratorA = typename ThreadblockMma::IteratorA;
  using IteratorB = typename ThreadblockMma::IteratorB;

  const int SZ_K = lhs_dim2;
  const int SZ_N = rhs_dim2;
  // todo(guray) It isn't accurate when M isn't divisable to TILE_N.
  const int SZ_M = gridDim.y * Tile_n;

  // Set entire matrix as the problem size
  cutlass::gemm::GemmCoord problem_size(SZ_M, SZ_N, SZ_K);

  // Dynamic shared memory base pointer
  extern __shared__ ElementC GemmSharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  typename ThreadblockMma::SharedStorage* shared_storage =
      reinterpret_cast<typename ThreadblockMma::SharedStorage*>(
          GemmSharedStorageBase);

  // Compute threadblock location
  cutlass::gemm::GemmCoord tb_tile_offset = {int(blockIdx.y), int(blockIdx.x),
                                             0};

  cutlass::MatrixCoord tb_offset_A{
      tb_tile_offset.m() * ThreadblockMma::Shape::kM, tb_tile_offset.k()};

  cutlass::MatrixCoord tb_offset_B{
      tb_tile_offset.k(), tb_tile_offset.n() * ThreadblockMma::Shape::kN};

  // Compute position within threadblock (linearized thread ID)
  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
  int lane_id = tb_thread_id & 0x1f;

  typename IteratorA::Params params_A(
      cutlass::layout::RowMajor::packed({problem_size.m(), problem_size.k()}));
  typename IteratorB::Params params_B(
      cutlass::layout::RowMajor::packed({problem_size.k(), problem_size.n()}));

  // Construct iterators to A and B operands
  typename ThreadblockMma::IteratorA iterator_A(
      params_A, plhs, {problem_size.m(), problem_size.k()}, tb_thread_id,
      tb_offset_A);

  typename ThreadblockMma::IteratorB iterator_B(
      params_B, prhs, {problem_size.k(), problem_size.n()}, tb_thread_id,
      tb_offset_B);

  // Construct thread-scoped matrix multiply
  ThreadblockMma mma(*shared_storage, tb_thread_id, warp_id, lane_id);

  typename ThreadblockMma::FragmentC accum;

  accum.clear();

  int gemm_k_iterations = (problem_size.k() + ThreadblockMma::Shape::kK - 1) /
                          ThreadblockMma::Shape::kK;

  // Compute threadblock-scoped matrix multiply-add
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

#ifdef DEBUG_CUTLASS
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nAll threads dump all the elements:\n");
  cutlass::debug::dump_fragment(accum);
#endif

  /* Use itself */
  typename ThreadblockMma::FragmentC accumC;
  if (!hasLinalgFill) {
    accumC.clear();
    iterator_C.load(accumC);
  }

  /* Store accumulator to shared memory */
  int total_elements = accum.size();
  ElementC* offset_shmem =
      &GemmSharedStorageBase[tb_thread_id * total_elements];
  for (int i = 0; i < total_elements; ++i)
    ElementC res =
        initValue +
        ElementC(typename ThreadblockMma::FragmentC::value_type(accum[i]));
  if (!hasLinalgFill) {
    res += accumC[i];
  }
  offset_shmem[i] = res;
}

#endif  // UCUDA_GEMM_CUTLASS
