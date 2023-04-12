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
! Copyright (c) 2013      Los Alamos National Security, LLC. All rights
!                         reserved.
! $COPYRIGHT$
!
! Additional copyrights may follow
!
! $HEADER$
!

!
!     This file contains the output from configure that is relevant for
!     Fortran applications and a few values that are necessary to
!     compile the Fortran modules (e.g., MPI_STATUS_SIZE).
!

!
!     OMPI version
!     This file is generated from configure; do not edit it manually.
!
      integer OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION
      integer OMPI_RELEASE_VERSION
      parameter (OMPI_MAJOR_VERSION=4)
      parameter (OMPI_MINOR_VERSION=1)
      parameter (OMPI_RELEASE_VERSION=4)
!
!     Kind parameters
!
      integer MPI_OFFSET_KIND, MPI_ADDRESS_KIND, MPI_INTEGER_KIND
      integer MPI_COUNT_KIND
      parameter (MPI_INTEGER_KIND=4)
      parameter (MPI_ADDRESS_KIND=8)
      parameter (MPI_OFFSET_KIND=8)
      parameter (MPI_COUNT_KIND=8)
!
!     Miscellaneous constants
!
      integer MPI_STATUS_SIZE
      parameter (MPI_STATUS_SIZE=6)
!
!     Configurable length constants
!
      integer MPI_MAX_PROCESSOR_NAME
      integer MPI_MAX_ERROR_STRING
      integer MPI_MAX_OBJECT_NAME
      integer MPI_MAX_LIBRARY_VERSION_STRING
      integer MPI_MAX_INFO_KEY
      integer MPI_MAX_INFO_VAL
      integer MPI_MAX_PORT_NAME
      integer MPI_MAX_DATAREP_STRING
      parameter (MPI_MAX_PROCESSOR_NAME=256-1)
      parameter (MPI_MAX_ERROR_STRING=256-1)
      parameter (MPI_MAX_OBJECT_NAME=64-1)
      parameter (MPI_MAX_LIBRARY_VERSION_STRING=256-1)
      parameter (MPI_MAX_INFO_KEY=36-1)
      parameter (MPI_MAX_INFO_VAL=256-1)
      parameter (MPI_MAX_PORT_NAME=1024-1)
      parameter (MPI_MAX_DATAREP_STRING=128-1)

!
! MPI F08 conformance
!
      logical MPI_SUBARRAYS_SUPPORTED
      logical MPI_ASYNC_PROTECTS_NONBLOCKING
      ! Hard-coded for .false. for now
      parameter (MPI_SUBARRAYS_SUPPORTED= .false.)
      ! Hard-coded for .false. for now
      parameter (MPI_ASYNC_PROTECTS_NONBLOCKING = .false.)

