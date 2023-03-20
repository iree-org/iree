// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UCUDA_GEMM_CUTLASS
#define UCUDA_GEMM_CUTLASS

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#ifndef DEBUG_CUTLASS
#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"
#endif

template <class ElementA, class ElementB, class ElementC, int Tile_m,
          int Tile_n, int Tile_k, int Warp_m, int Warp_n, int Inst_m,
          int Inst_n, int Inst_k, int Stages, bool hasLinalgFill,
          bool writeBack2Global>
__forceinline__ __device__ void gemm_ukernel(
    ElementA* lhs, int64_t lhs_offset, int64_t lhs_dim2, ElementB* rhs,
    int64_t rhs_offset, int64_t rhs_dim2, ElementC* res, int64_t res_offset,
    int64_t res_dim2, ElementC* shmem, ElementC fillValue) {
  using ElementAccumulator = ElementC;
  // todo(guray) Can be templatized
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ThreadblockShape = cutlass::gemm::GemmShape<Tile_m, Tile_n, Tile_k>;
  using WarpShape = cutlass::gemm::GemmShape<Warp_m, Warp_n, Tile_k>;
  using InstructionShape = cutlass::gemm::GemmShape<Inst_m, Inst_n, Inst_k>;

  int const kAlignmentA = 4;
  int const kAlignmentB = 4;

  // CUTLASS Threadblock-level matmul operator and global memory tile
  // iterators
  using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape,
      Stages, cutlass::arch::OpMultiplyAdd>;

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
  
  const int split_k_slices = 1;

  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; 
  ThreadblockSwizzle threadblock_swizzle;

  cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
      problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK}, split_k_slices);

  int swizzle_log_tile = ThreadblockSwizzle().get_log_tile(grid_tiled_shape);

  // Compute threadblock location
  cutlass::gemm::GemmCoord tb_tile_offset = threadblock_swizzle.get_tile_offset(swizzle_log_tile);

  // Dynamic shared memory base pointer
  extern __shared__ ElementC GemmSharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  typename ThreadblockMma::SharedStorage* shared_storage =
      reinterpret_cast<typename ThreadblockMma::SharedStorage*>(
          GemmSharedStorageBase);


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
      params_A, lhs, {problem_size.m(), problem_size.k()}, tb_thread_id,
      tb_offset_A);

  typename ThreadblockMma::IteratorB iterator_B(
      params_B, rhs, {problem_size.k(), problem_size.n()}, tb_thread_id,
      tb_offset_B);

  typename ThreadblockMma::Operator::IteratorC iterator_LoadC(
      {res, problem_size.n()}, threadIdx.x);

  // Construct thread-scoped matrix multiply
  ThreadblockMma mma(*shared_storage, tb_thread_id, warp_id, lane_id);

  typename ThreadblockMma::FragmentC accumSrc, accumDest;
  accumDest.clear();

  int gemm_k_iterations = (problem_size.k() + ThreadblockMma::Shape::kK - 1) /
                          ThreadblockMma::Shape::kK;

  if (!hasLinalgFill) {
    // Set the offset
    iterator_LoadC.add_tile_offset(
        {(tb_tile_offset.m() * ThreadblockMma::WarpCount::kM) +
             (warp_id % ThreadblockMma::WarpCount::kM),
         (tb_tile_offset.n() * ThreadblockMma::WarpCount::kN) +
             (warp_id / ThreadblockMma::WarpCount::kM)});

    // Clear the fragment
    accumSrc.clear();

    // Load C as source accumulator
    iterator_LoadC.load(accumSrc);

    // Compute threadblock-scoped matrix multiply-add
    mma(gemm_k_iterations, accumDest, iterator_A, iterator_B, accumSrc);
  } else {
    // Compute threadblock-scoped matrix multiply-add
    mma(gemm_k_iterations, accumDest, iterator_A, iterator_B, accumDest);
  }

#ifdef DEBUG_CUTLASS
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nAll threads dump all the elements:\n");
  cutlass::debug::dump_fragment(accum);
#endif

  if (!writeBack2Global) {
    /* Store result to shared memory */
    int total_elements = accumDest.size();
    ElementC* offset_shmem =
        &GemmSharedStorageBase[tb_thread_id * total_elements];
    for (int i = 0; i < accumDest.size(); ++i) {
      offset_shmem[i] = accumDest[i];
      if (hasLinalgFill) offset_shmem[i] += fillValue;
    }
  } else {
    int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementC>::value;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, kElementsPerAccess, ElementC, ElementC>;

    using Epilogue =
        typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
            ThreadblockShape, typename ThreadblockMma::Operator,
            ThreadblockMma::Policy::kPartitionsK, EpilogueOutputOp,
            EpilogueOutputOp::kCount>::Epilogue;

    tb_tile_offset = threadblock_swizzle.get_tile_offset(swizzle_log_tile);

    // assume identity swizzle
    cutlass::MatrixCoord threadblock_offset{
        tb_tile_offset.m() * ThreadblockMma::Shape::kM,
        tb_tile_offset.n() * ThreadblockMma::Shape::kN};

    // Create Layout
    typename Epilogue::OutputTileIterator::Params params_D(
        cutlass::layout::RowMajor::packed(
            {problem_size.m(), problem_size.n()}));

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_D(
        params_D, res, problem_size.mn(), tb_thread_id, threadblock_offset);

    typename EpilogueOutputOp::Params params_output_op;

    EpilogueOutputOp output_op(params_output_op);

    // Reuse the same shared memory that we used for the inputs
    typename Epilogue::SharedStorage* e_shared_storage =
        reinterpret_cast<typename Epilogue::SharedStorage*>(
            GemmSharedStorageBase);

    Epilogue epilogue(*e_shared_storage, tb_thread_id, warp_id, lane_id);

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op, iterator_D, accumDest);
  }
}

#endif  // UCUDA_GEMM_CUTLASS
