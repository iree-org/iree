// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

IREE_NCCL_PFN_DECL(ncclGetVersion, int*)
IREE_NCCL_PFN_DECL(ncclGetUniqueId, ncclUniqueId*)
IREE_NCCL_PFN_DECL(ncclCommInitRankConfig, ncclComm_t*, int, ncclUniqueId, int,
                   ncclConfig_t*)
IREE_NCCL_PFN_DECL(ncclCommInitRank, ncclComm_t*, int, ncclUniqueId, int)
IREE_NCCL_PFN_DECL(ncclCommInitAll, ncclComm_t*, int, const int*)
IREE_NCCL_PFN_DECL(ncclCommSplit, ncclComm_t, int, int, ncclComm_t*,
                   ncclConfig_t*)
IREE_NCCL_PFN_DECL(ncclCommFinalize, ncclComm_t)
IREE_NCCL_PFN_DECL(ncclCommDestroy, ncclComm_t)
IREE_NCCL_PFN_DECL(ncclCommAbort, ncclComm_t)
IREE_NCCL_PFN_DECL_STR_RETURN(ncclGetErrorString, ncclResult_t)
IREE_NCCL_PFN_DECL_STR_RETURN(ncclGetLastError, ncclComm_t)
IREE_NCCL_PFN_DECL(ncclCommGetAsyncError, ncclComm_t, ncclResult_t*)
IREE_NCCL_PFN_DECL(ncclCommCount, const ncclComm_t, int*)
IREE_NCCL_PFN_DECL(ncclCommCuDevice, const ncclComm_t, int*)
IREE_NCCL_PFN_DECL(ncclCommUserRank, const ncclComm_t, int*)
IREE_NCCL_PFN_DECL(ncclRedOpCreatePreMulSum, ncclRedOp_t*, void*,
                   ncclDataType_t, ncclScalarResidence_t, ncclComm_t)
IREE_NCCL_PFN_DECL(ncclRedOpDestroy, ncclRedOp_t, ncclComm_t)
IREE_NCCL_PFN_DECL(ncclReduce, const void*, void*, size_t, ncclDataType_t,
                   ncclRedOp_t, int, ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclBcast, void*, size_t, ncclDataType_t, int, ncclComm_t,
                   cudaStream_t)
IREE_NCCL_PFN_DECL(ncclBroadcast, const void*, void*, size_t, ncclDataType_t,
                   int, ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclAllReduce, const void*, void*, size_t, ncclDataType_t,
                   ncclRedOp_t, ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclReduceScatter, const void*, void*, size_t,
                   ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclAllGather, const void*, void*, size_t, ncclDataType_t,
                   ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclSend, const void*, size_t, ncclDataType_t, int,
                   ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclRecv, void*, size_t, ncclDataType_t, int, ncclComm_t,
                   cudaStream_t)
IREE_NCCL_PFN_DECL(ncclGroupStart)
IREE_NCCL_PFN_DECL(ncclGroupEnd)
