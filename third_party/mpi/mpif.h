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
! Copyright (c) 2006-2014 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2017      Research Organization for Information Science
!                         and Technology (RIST). All rights reserved.
! $COPYRIGHT$
!
! Additional copyrights may follow
!
! $HEADER$
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! Do ***not*** copy this file to the directory where your Fortran
! fortran application is compiled unless it is absolutely necessary!  Most
! modern Fortran compilers now support the -I command line flag, which
! tells the compiler where to find .h files (specifically, this one).  For
! example:
!
!      shell$ mpifort foo.f -o foo -I$OMPI_HOME/include
!
! will probably do the trick (assuming that you have set OMPI_HOME
! properly).
!
! That being said, OMPI's "mpifort" wrapper compiler should
! automatically include the -I option for you.  The following command
! should be equivalent to the command listed above:
!
!      shell$ mpifort foo.f -o foo
!
! You should not copy this file to your local directory because it is
! possible that this file will be changed between versions of Open MPI.
! Indeed, this mpif.h is incompatible with the mpif.f of other
! implementations of MPI.  Using this mpif.h with other implementations
! of MPI, or with other versions of Open MPI will result in undefined
! behavior (to include incorrect results, segmentation faults,
! unexplainable "hanging" in your application, etc.).  Always use the
! -I command line option instead (or let mpifort do it for you).
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      include 'mpif-config.h'
      include 'mpif-constants.h'
      include 'mpif-handles.h'
      include 'mpif-io-constants.h'
      include 'mpif-io-handles.h'
      include 'mpif-externals.h'
      include 'mpif-sentinels.h'
      include 'mpif-sizeof.h'
