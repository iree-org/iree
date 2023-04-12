! -*- fortran -*-
! WARNING! THIS IS A GENERATED FILE!!
! ANY EDITS YOU PUT HERE WILL BE LOST!
! ==> Instead, edit topdir/ompi/include/mpif-values.pl.

! Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
!                         University Research and Technology
!                         Corporation.  All rights reserved.
! Copyright (c) 2004-2010 The University of Tennessee and The University
!                         of Tennessee Research Foundation.  All rights
!                         reserved.
! Copyright (c) 2004-2007 High Performance Computing Center Stuttgart,
!                         University of Stuttgart.  All rights reserved.
! Copyright (c) 2004-2005 The Regents of the University of California.
!                         All rights reserved.
! Copyright (c) 2006-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
! Copyright (c) 2016      Research Organization for Information Science
!                         and Technology (RIST). All rights reserved.
! $COPYRIGHT$
!
! Additional copyrights may follow
!
! $HEADER$
!

        integer MPI_2COMPLEX
        integer MPI_2DOUBLE_COMPLEX
        integer MPI_2DOUBLE_PRECISION
        integer MPI_2INT
        integer MPI_2INTEGER
        integer MPI_2REAL
        integer MPI_AINT
        integer MPI_BAND
        integer MPI_BOR
        integer MPI_BXOR
        integer MPI_BYTE
        integer MPI_CHAR
        integer MPI_CHARACTER
        integer MPI_COMM_NULL
        integer MPI_COMM_SELF
        integer MPI_COMM_WORLD
        integer MPI_COMPLEX
        integer MPI_COMPLEX16
        integer MPI_COMPLEX32
        integer MPI_COMPLEX8
        integer MPI_COUNT
        integer MPI_CXX_BOOL
        integer MPI_CXX_COMPLEX
        integer MPI_CXX_DOUBLE_COMPLEX
        integer MPI_CXX_FLOAT_COMPLEX
        integer MPI_CXX_LONG_DOUBLE_COMPLEX
        integer MPI_C_BOOL
        integer MPI_C_COMPLEX
        integer MPI_C_DOUBLE_COMPLEX
        integer MPI_C_FLOAT_COMPLEX
        integer MPI_C_LONG_DOUBLE_COMPLEX
        integer MPI_DATATYPE_NULL
        integer MPI_DOUBLE
        integer MPI_DOUBLE_COMPLEX
        integer MPI_DOUBLE_INT
        integer MPI_DOUBLE_PRECISION
        integer MPI_ERRHANDLER_NULL
        integer MPI_ERRORS_ARE_FATAL
        integer MPI_ERRORS_RETURN
        integer MPI_FLOAT
        integer MPI_FLOAT_INT
        integer MPI_GROUP_EMPTY
        integer MPI_GROUP_NULL
        integer MPI_INFO_ENV
        integer MPI_INFO_NULL
        integer MPI_INT
        integer MPI_INT16_T
        integer MPI_INT32_T
        integer MPI_INT64_T
        integer MPI_INT8_T
        integer MPI_INTEGER
        integer MPI_INTEGER1
        integer MPI_INTEGER16
        integer MPI_INTEGER2
        integer MPI_INTEGER4
        integer MPI_INTEGER8
        integer MPI_LAND
        integer MPI_LB
        integer MPI_LOGICAL
        integer MPI_LOGICAL1
        integer MPI_LOGICAL2
        integer MPI_LOGICAL4
        integer MPI_LOGICAL8
        integer MPI_LONG
        integer MPI_LONG_DOUBLE
        integer MPI_LONG_DOUBLE_INT
        integer MPI_LONG_INT
        integer MPI_LONG_LONG
        integer MPI_LONG_LONG_INT
        integer MPI_LOR
        integer MPI_LXOR
        integer MPI_MAX
        integer MPI_MAXLOC
        integer MPI_MESSAGE_NO_PROC
        integer MPI_MESSAGE_NULL
        integer MPI_MIN
        integer MPI_MINLOC
        integer MPI_NO_OP
        integer MPI_OFFSET
        integer MPI_OP_NULL
        integer MPI_PACKED
        integer MPI_PROD
        integer MPI_REAL
        integer MPI_REAL16
        integer MPI_REAL2
        integer MPI_REAL4
        integer MPI_REAL8
        integer MPI_REPLACE
        integer MPI_REQUEST_NULL
        integer MPI_SHORT
        integer MPI_SHORT_INT
        integer MPI_SIGNED_CHAR
        integer MPI_SUM
        integer MPI_UB
        integer MPI_UINT16_T
        integer MPI_UINT32_T
        integer MPI_UINT64_T
        integer MPI_UINT8_T
        integer MPI_UNSIGNED
        integer MPI_UNSIGNED_CHAR
        integer MPI_UNSIGNED_LONG
        integer MPI_UNSIGNED_LONG_LONG
        integer MPI_UNSIGNED_SHORT
        integer MPI_WCHAR
        integer MPI_WIN_NULL

        parameter (MPI_2COMPLEX=26)
        parameter (MPI_2DOUBLE_COMPLEX=27)
        parameter (MPI_2DOUBLE_PRECISION=24)
        parameter (MPI_2INT=52)
        parameter (MPI_2INTEGER=25)
        parameter (MPI_2REAL=23)
        parameter (MPI_AINT=66)
        parameter (MPI_BAND=6)
        parameter (MPI_BOR=8)
        parameter (MPI_BXOR=10)
        parameter (MPI_BYTE=1)
        parameter (MPI_CHAR=34)
        parameter (MPI_CHARACTER=5)
        parameter (MPI_COMM_NULL=2)
        parameter (MPI_COMM_SELF=1)
        parameter (MPI_COMM_WORLD=0)
        parameter (MPI_COMPLEX=18)
        parameter (MPI_COMPLEX16=20)
        parameter (MPI_COMPLEX32=21)
        parameter (MPI_COMPLEX8=19)
        parameter (MPI_COUNT=72)
        parameter (MPI_CXX_BOOL=54)
        parameter (MPI_CXX_COMPLEX=55)
        parameter (MPI_CXX_DOUBLE_COMPLEX=56)
        parameter (MPI_CXX_FLOAT_COMPLEX=55)
        parameter (MPI_CXX_LONG_DOUBLE_COMPLEX=57)
        parameter (MPI_C_BOOL=68)
        parameter (MPI_C_COMPLEX=69)
        parameter (MPI_C_DOUBLE_COMPLEX=70)
        parameter (MPI_C_FLOAT_COMPLEX=69)
        parameter (MPI_C_LONG_DOUBLE_COMPLEX=71)
        parameter (MPI_DATATYPE_NULL=0)
        parameter (MPI_DOUBLE=46)
        parameter (MPI_DOUBLE_COMPLEX=22)
        parameter (MPI_DOUBLE_INT=49)
        parameter (MPI_DOUBLE_PRECISION=17)
        parameter (MPI_ERRHANDLER_NULL=0)
        parameter (MPI_ERRORS_ARE_FATAL=1)
        parameter (MPI_ERRORS_RETURN=2)
        parameter (MPI_FLOAT=45)
        parameter (MPI_FLOAT_INT=48)
        parameter (MPI_GROUP_EMPTY=1)
        parameter (MPI_GROUP_NULL=0)
        parameter (MPI_INFO_ENV=1)
        parameter (MPI_INFO_NULL=0)
        parameter (MPI_INT=39)
        parameter (MPI_INT16_T=60)
        parameter (MPI_INT32_T=62)
        parameter (MPI_INT64_T=64)
        parameter (MPI_INT8_T=58)
        parameter (MPI_INTEGER=7)
        parameter (MPI_INTEGER1=8)
        parameter (MPI_INTEGER16=12)
        parameter (MPI_INTEGER2=9)
        parameter (MPI_INTEGER4=10)
        parameter (MPI_INTEGER8=11)
        parameter (MPI_LAND=5)
        parameter (MPI_LB=4)
        parameter (MPI_LOGICAL=6)
        parameter (MPI_LOGICAL1=29)
        parameter (MPI_LOGICAL2=30)
        parameter (MPI_LOGICAL4=31)
        parameter (MPI_LOGICAL8=32)
        parameter (MPI_LONG=41)
        parameter (MPI_LONG_DOUBLE=47)
        parameter (MPI_LONG_DOUBLE_INT=50)
        parameter (MPI_LONG_INT=51)
        parameter (MPI_LONG_LONG=43)
        parameter (MPI_LONG_LONG_INT=43)
        parameter (MPI_LOR=7)
        parameter (MPI_LXOR=9)
        parameter (MPI_MAX=1)
        parameter (MPI_MAXLOC=11)
        parameter (MPI_MESSAGE_NO_PROC=1)
        parameter (MPI_MESSAGE_NULL=0)
        parameter (MPI_MIN=2)
        parameter (MPI_MINLOC=12)
        parameter (MPI_NO_OP=14)
        parameter (MPI_OFFSET=67)
        parameter (MPI_OP_NULL=0)
        parameter (MPI_PACKED=2)
        parameter (MPI_PROD=4)
        parameter (MPI_REAL=13)
        parameter (MPI_REAL16=16)
        parameter (MPI_REAL2=28)
        parameter (MPI_REAL4=14)
        parameter (MPI_REAL8=15)
        parameter (MPI_REPLACE=13)
        parameter (MPI_REQUEST_NULL=0)
        parameter (MPI_SHORT=37)
        parameter (MPI_SHORT_INT=53)
        parameter (MPI_SIGNED_CHAR=36)
        parameter (MPI_SUM=3)
        parameter (MPI_UB=3)
        parameter (MPI_UINT16_T=61)
        parameter (MPI_UINT32_T=63)
        parameter (MPI_UINT64_T=65)
        parameter (MPI_UINT8_T=59)
        parameter (MPI_UNSIGNED=40)
        parameter (MPI_UNSIGNED_CHAR=35)
        parameter (MPI_UNSIGNED_LONG=42)
        parameter (MPI_UNSIGNED_LONG_LONG=44)
        parameter (MPI_UNSIGNED_SHORT=38)
        parameter (MPI_WCHAR=33)
        parameter (MPI_WIN_NULL=0)
