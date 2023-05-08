// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

MPI_PFN_DECL(MPI_Init, int*, char***)
MPI_PFN_DECL(MPI_Initialized, int*)
MPI_PFN_DECL(MPI_Finalize)
MPI_PFN_DECL(MPI_Bcast, void* buffer, int count, IREE_MPI_Datatype datatype,
             int root, IREE_MPI_Comm comm)
MPI_PFN_DECL(MPI_Comm_rank, IREE_MPI_Comm comm, int* rank)
MPI_PFN_DECL(MPI_Comm_size, IREE_MPI_Comm comm, int* size)
MPI_PFN_DECL(MPI_Comm_split, IREE_MPI_Comm comm, int color, int key,
             IREE_MPI_Comm* newcomm)

#if IREE_MPI_TYPES_ARE_POINTERS
MPI_PFN_DECL(ompi_mpi_byte)
MPI_PFN_DECL(ompi_mpi_comm_world)
#endif  // IREE_MPI_TYPES_ARE_POINTERS
