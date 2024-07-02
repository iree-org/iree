// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

MPI_PFN_DECL(MPI_Init, int*, char***)
MPI_PFN_DECL(MPI_Initialized, int*)
MPI_PFN_DECL(MPI_Finalize)
MPI_PFN_DECL(MPI_Comm_rank, IREE_MPI_Comm comm, int* rank)
MPI_PFN_DECL(MPI_Comm_size, IREE_MPI_Comm comm, int* size)

// MPI error handling
MPI_PFN_DECL(MPI_Error_class, int err_code, int* err_class)
MPI_PFN_DECL(MPI_Error_string, int err_code, char* string, int* resultlen)

// MPI collectives
MPI_PFN_DECL(MPI_Bcast, void* buffer, int count, IREE_MPI_Datatype datatype,
             int root, IREE_MPI_Comm comm)
MPI_PFN_DECL(MPI_Comm_split, IREE_MPI_Comm comm, int color, int key,
             IREE_MPI_Comm* newcomm)
MPI_PFN_DECL(MPI_Gather, void* sendbuf, int sendcount,
             IREE_MPI_Datatype sendtype, void* recvbuf, int recvcount,
             IREE_MPI_Datatype recvtype, int root, IREE_MPI_Comm comm)
MPI_PFN_DECL(MPI_Scatter, void* sendbuf, int sendcount,
             IREE_MPI_Datatype sendtype, void* recvbuf, int recvcount,
             IREE_MPI_Datatype recvtype, int root, IREE_MPI_Comm comm)
MPI_PFN_DECL(MPI_Allgather, void* sendbuf, int sendcount,
             IREE_MPI_Datatype sendtype, void* recvbuf, int recvcount,
             IREE_MPI_Datatype recvtype, IREE_MPI_Comm comm)
MPI_PFN_DECL(MPI_Alltoall, void* sendbuf, int sendcount,
             IREE_MPI_Datatype sendtype, void* recvbuf, int recvcount,
             IREE_MPI_Datatype recvtype, IREE_MPI_Comm comm)
MPI_PFN_DECL(MPI_Reduce, void* sendbuf, void* recvbuf, int count,
             IREE_MPI_Datatype datatype, IREE_MPI_Op op, int root,
             IREE_MPI_Comm comm)
MPI_PFN_DECL(MPI_Allreduce, void* sendbuf, void* recvbuf, int count,
             IREE_MPI_Datatype datatype, IREE_MPI_Op op, IREE_MPI_Comm comm)

#if IREE_MPI_TYPES_ARE_POINTERS
MPI_PFN_DECL(ompi_mpi_comm_world)

// MPI data type handles
MPI_PFN_DECL(ompi_mpi_byte)
MPI_PFN_DECL(ompi_mpi_int)
MPI_PFN_DECL(ompi_mpi_float)
MPI_PFN_DECL(ompi_mpi_double)

// MPI op handles
MPI_PFN_DECL(ompi_mpi_op_max)
MPI_PFN_DECL(ompi_mpi_op_min)
MPI_PFN_DECL(ompi_mpi_op_sum)
MPI_PFN_DECL(ompi_mpi_op_prod)
MPI_PFN_DECL(ompi_mpi_op_land)
MPI_PFN_DECL(ompi_mpi_op_band)
MPI_PFN_DECL(ompi_mpi_op_lor)
MPI_PFN_DECL(ompi_mpi_op_bor)
MPI_PFN_DECL(ompi_mpi_op_lxor)
MPI_PFN_DECL(ompi_mpi_op_bxor)
MPI_PFN_DECL(ompi_mpi_op_maxloc)
MPI_PFN_DECL(ompi_mpi_op_minloc)
MPI_PFN_DECL(ompi_mpi_op_replace)
MPI_PFN_DECL(ompi_mpi_op_no_op)
#endif  // IREE_MPI_TYPES_ARE_POINTERS
