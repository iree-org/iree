! -*- fortran -*-
!
! Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
!                         University Research and Technology
!                         Corporation.  All rights reserved.
! Copyright (c) 2004-2005 The University of Tennessee and The University
!                         of Tennessee Research Foundation.  All rights
!                         reserved.
! Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
!                         University of Stuttgart.  All rights reserved.
! Copyright (c) 2004-2005 The Regents of the University of California.
!                         All rights reserved.
! Copyright (c) 2006-2017 Cisco Systems, Inc.  All rights reserved
! $COPYRIGHT$
!
! Additional copyrights may follow
!
! $HEADER$
!

!
!     These "external" statements are specific to the MPI mpif.h
!     interface (and are toxic to the MPI module interfaces),.
!
      external MPI_NULL_COPY_FN, MPI_NULL_DELETE_FN
      external MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN
      external MPI_TYPE_NULL_COPY_FN, MPI_TYPE_NULL_DELETE_FN
      external MPI_DUP_FN, MPI_COMM_DUP_FN, MPI_TYPE_DUP_FN
      external MPI_WIN_NULL_COPY_FN
      external MPI_WIN_NULL_DELETE_FN
      external MPI_WIN_DUP_FN
!     Note that MPI_CONVERSION_FN_NULL is a "constant" (it is only ever
!     checked for comparison; it is never invoked), but it is passed as
!     a function pointer (to MPI_REGISTER_DATAREP) and therefore must be
!     the same size/type.  It is therefore external'ed here, and not
!     defined with an integer value in mpif-common.h.
      external MPI_CONVERSION_FN_NULL

!
!     double precision functions
!
      external MPI_WTIME, MPI_WTICK , PMPI_WTICK, PMPI_WTIME
      double precision MPI_WTIME, MPI_WTICK , PMPI_WTICK, PMPI_WTIME
!
!     address integer functions
!
      external MPI_AINT_ADD, MPI_AINT_DIFF
      integer(kind=MPI_ADDRESS_KIND) MPI_AINT_ADD, MPI_AINT_DIFF
