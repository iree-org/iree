// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MPI dynamic symbols
MPI_PFN_DECL(MPI_Init, int*, char***)
MPI_PFN_DECL(MPI_Finalize)
MPI_PFN_DECL(MPI_Bcast, void* buffer, int count, void* datatype, int root,
             void* comm)
MPI_PFN_DECL(MPI_Comm_rank, void* comm, int* rank)
MPI_PFN_DECL(MPI_Comm_size, void* comm, int* size)
MPI_PFN_DECL(MPI_Comm_split, void* comm, int color, int key, void** newcomm)
MPI_PFN_DECL(ompi_mpi_byte)
MPI_PFN_DECL(ompi_mpi_comm_world)
