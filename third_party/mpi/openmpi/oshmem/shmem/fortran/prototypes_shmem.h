/*
 * Copyright (c) 2014      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OSHMEM_F77_PROTOTYPES_SHMEM_H
#define OSHMEM_F77_PROTOTYPES_SHMEM_H
#include "oshmem_config.h"
#include "shmem_fortran_pointer.h"

BEGIN_C_DECLS

#define PN(ret, lower_name, upper_name, args) \
  OSHMEM_DECLSPEC ret lower_name##_f args;    \
  OSHMEM_DECLSPEC ret lower_name##_ args;     \
  OSHMEM_DECLSPEC ret lower_name##__ args;    \
  OSHMEM_DECLSPEC ret upper_name args

PN (void, shmem_init, SHMEM_INIT, (void));
PN (void, start_pes, START_PES, (MPI_Fint npes));
PN (void, shmem_global_exit, SHMEM_GLOBAL_EXIT, (MPI_Fint npes));
PN (MPI_Fint, shmem_n_pes, SHMEM_N_PES, (void));
PN (MPI_Fint, num_pes, NUM_PES, (void));
PN (MPI_Fint, shmem_my_pe, SHMEM_MY_PE, (void));
PN (MPI_Fint, my_pe, MY_PE, (void));
OSHMEM_DECLSPEC MPI_Fint _my_pe_(void);
PN (void, shmem_info_get_version, SHMEM_INFO_GET_VERSION, (MPI_Fint *major, MPI_Fint *minor));
PN (void, shmem_info_get_name, SHMEM_INFO_GET_NAME, (char *name));
PN (void, shmem_finalize, SHMEM_FINALIZE, (void));
PN (void, shmem_barrier_all, SHMEM_BARRIER_ALL, (void));
PN (void, shpalloc, SHPALLOC, (FORTRAN_POINTER_T *addr, MPI_Fint *length, MPI_Fint *errcode, MPI_Fint *abort));
PN (void, shpdeallc, SHPDEALLC, (FORTRAN_POINTER_T *addr, MPI_Fint *errcode, MPI_Fint *abort));
PN (void, shpclmove, SHPCLMOVE, (FORTRAN_POINTER_T *addr, MPI_Fint *length, MPI_Fint *status, MPI_Fint *abort));
PN (FORTRAN_POINTER_T*, shmem_ptr, SHMEM_PTR, (FORTRAN_POINTER_T target, MPI_Fint *pe));
PN (ompi_fortran_logical_t, shmem_pe_accessible, SHMEM_PE_ACCESSIBLE, (MPI_Fint *pe));
PN (MPI_Fint, shmem_addr_accessible, SHMEM_ADDR_ACCESSIBLE, (FORTRAN_POINTER_T addr, MPI_Fint *pe));

PN (void, shmem_put, SHMEM_PUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_character_put, SHMEM_CHARACTER_PUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_complex_put, SHMEM_COMPLEX_PUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_double_put, SHMEM_DOUBLE_PUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_logical_put, SHMEM_LOGICAL_PUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_integer_put, SHMEM_INTEGER_PUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_real_put, SHMEM_REAL_PUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_put4, SHMEM_PUT4, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_put8, SHMEM_PUT8, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_put32, SHMEM_PUT32, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_put64, SHMEM_PUT64, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_put128, SHMEM_PUT128, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_putmem, SHMEM_PUTMEM, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));

PN (void, shmem_iput4, SHMEM_IPUT4, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_iput8, SHMEM_IPUT8, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_iput32, SHMEM_IPUT32, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_iput64, SHMEM_IPUT64, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_iput128, SHMEM_IPUT128, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_complex_iput, SHMEM_COMPLEX_IPUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_double_iput, SHMEM_DOUBLE_IPUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_integer_iput, SHMEM_INTEGER_IPUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_logical_iput, SHMEM_LOGICAL_IPUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_real_iput, SHMEM_REAL_IPUT, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));

PN (void, shmem_putmem_nbi, SHMEM_PUTMEM_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_character_put_nbi, SHMEM_CHARACTER_PUT_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_complex_put_nbi, SHMEM_COMPLEX_PUT_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_double_put_nbi, SHMEM_DOUBLE_PUT_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_integer_put_nbi, SHMEM_INTEGER_PUT_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_logical_put_nbi, SHMEM_LOGICAL_PUT_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_real_put_nbi, SHMEM_REAL_PUT_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_put4_nbi, SHMEM_PUT4_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_put8_nbi, SHMEM_PUT8_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_put32_nbi, SHMEM_PUT32_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_put64_nbi, SHMEM_PUT64_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));
PN (void, shmem_put128_nbi, SHMEM_PUT128_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe));

PN (void, shmem_character_get, SHMEM_CHARACTER_GET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_complex_get, SHMEM_COMPLEX_GET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_double_get, SHMEM_DOUBLE_GET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_integer_get, SHMEM_INTEGER_GET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_get4, SHMEM_GET4, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_get8, SHMEM_GET8, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_get32, SHMEM_GET32, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_get64, SHMEM_GET64, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_get128, SHMEM_GET128, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_getmem, SHMEM_GETMEM, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_logical_get, SHMEM_LOGICAL_GET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_real_get, SHMEM_REAL_GET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));

PN (void, shmem_iget4, SHMEM_IGET4, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_iget8, SHMEM_IGET8, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_iget32, SHMEM_IGET32, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_iget64, SHMEM_IGET64, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_iget128, SHMEM_IGET128, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_complex_iget, SHMEM_COMPLEX_IGET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_double_iget, SHMEM_DOUBLE_IGET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_integer_iget, SHMEM_INTEGER_IGET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_logical_iget, SHMEM_LOGICAL_IGET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_real_iget, SHMEM_REAL_IGET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe));

PN (void, shmem_getmem_nbi, SHMEM_GETMEM_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_character_get_nbi, SHMEM_CHARACTER_GET_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_complex_get_nbi, SHMEM_COMPLEX_GET_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_double_get_nbi, SHMEM_DOUBLE_GET_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_integer_get_nbi, SHMEM_INTEGER_GET_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_logical_get_nbi, SHMEM_LOGICAL_GET_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_real_get_nbi, SHMEM_REAL_GET_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_get4_nbi, SHMEM_GET4_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_get8_nbi, SHMEM_GET8_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_get32_nbi, SHMEM_GET32_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_get64_nbi, SHMEM_GET64_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));
PN (void, shmem_get128_nbi, SHMEM_GET128_NBI, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *len, MPI_Fint *pe));


PN (MPI_Fint, shmem_swap, SHMEM_SWAP, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));
PN (ompi_fortran_integer4_t, shmem_int4_swap, SHMEM_INT4_SWAP, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));
PN (ompi_fortran_integer8_t, shmem_int8_swap, SHMEM_INT8_SWAP, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));
PN (ompi_fortran_real4_t, shmem_real4_swap, SHMEM_REAL4_SWAP, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));
PN (ompi_fortran_real8_t, shmem_real8_swap, SHMEM_REAL8_SWAP, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));

PN (void, shmem_int4_set, SHMEM_INT4_SET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));
PN (void, shmem_int8_set, SHMEM_INT8_SET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));
PN (void, shmem_real4_set, SHMEM_REAL4_SET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));
PN (void, shmem_real8_set, SHMEM_REAL8_SET, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));

PN (ompi_fortran_integer4_t, shmem_int4_cswap, SHMEM_INT4_CSWAP, (FORTRAN_POINTER_T target, MPI_Fint *cond, FORTRAN_POINTER_T value, MPI_Fint *pe));
PN (ompi_fortran_integer8_t, shmem_int8_cswap, SHMEM_INT8_CSWAP, (FORTRAN_POINTER_T target, MPI_Fint *cond, FORTRAN_POINTER_T value, MPI_Fint *pe));

PN (ompi_fortran_integer4_t, shmem_int4_fadd, SHMEM_INT4_FADD, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));
PN (ompi_fortran_integer8_t, shmem_int8_fadd, SHMEM_INT8_FADD, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));

PN (ompi_fortran_integer4_t, shmem_int4_fetch, SHMEM_INT4_FETCH, (FORTRAN_POINTER_T target, MPI_Fint *pe));
PN (ompi_fortran_integer8_t, shmem_int8_fetch, SHMEM_INT8_FETCH, (FORTRAN_POINTER_T target, MPI_Fint *pe));
PN (ompi_fortran_real4_t, shmem_real4_fetch, SHMEM_REAL4_FETCH, (FORTRAN_POINTER_T target, MPI_Fint *pe));
PN (ompi_fortran_real8_t, shmem_real8_fetch, SHMEM_REAL8_FETCH, (FORTRAN_POINTER_T target, MPI_Fint *pe));

PN (void, shmem_int4_inc, SHMEM_INT4_INC, (FORTRAN_POINTER_T target, MPI_Fint *pe));
PN (void, shmem_int8_inc, SHMEM_INT8_INC, (FORTRAN_POINTER_T target, MPI_Fint *pe));
PN (ompi_fortran_integer4_t, shmem_int4_finc, SHMEM_INT4_FINC, (FORTRAN_POINTER_T target, MPI_Fint *pe));
PN (ompi_fortran_integer8_t, shmem_int8_finc, SHMEM_INT8_FINC, (FORTRAN_POINTER_T target, MPI_Fint *pe));
PN (void, shmem_int4_add, SHMEM_INT4_ADD, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));
PN (void, shmem_int8_add, SHMEM_INT8_ADD, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T value, MPI_Fint *pe));
PN (void, shmem_int4_wait, SHMEM_INT4_WAIT, (ompi_fortran_integer4_t *var, ompi_fortran_integer4_t *value));
PN (void, shmem_int8_wait, SHMEM_INT8_WAIT, (ompi_fortran_integer8_t *var, ompi_fortran_integer8_t *value));
PN (void, shmem_wait, SHMEM_WAIT, (MPI_Fint *var, MPI_Fint *value));
PN (void, shmem_int4_wait_until, SHMEM_INT4_WAIT_UNTIL, (ompi_fortran_integer4_t *var, MPI_Fint *cmp, ompi_fortran_integer4_t *value));
PN (void, shmem_int8_wait_until, SHMEM_INT8_WAIT_UNTIL, (ompi_fortran_integer8_t *var, MPI_Fint *cmp, ompi_fortran_integer8_t *value));
PN (void, shmem_wait_until, SHMEM_WAIT_UNTIL, (MPI_Fint *var, MPI_Fint *cmp, MPI_Fint *value));
PN (void, shmem_barrier, SHMEM_BARRIER, (MPI_Fint *PE_start, MPI_Fint *logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_int2_and_to_all, SHMEM_INT2_AND_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int4_and_to_all, SHMEM_INT4_AND_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int8_and_to_all, SHMEM_INT8_AND_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int2_or_to_all, SHMEM_INT2_OR_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int4_or_to_all, SHMEM_INT4_OR_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int8_or_to_all, SHMEM_INT8_OR_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int2_xor_to_all, SHMEM_INT2_XOR_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int4_xor_to_all, SHMEM_INT4_XOR_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int8_xor_to_all, SHMEM_INT8_XOR_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_comp4_xor_to_all, SHMEM_COMP4_XOR_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_comp8_xor_to_all, SHMEM_COMP8_XOR_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int2_max_to_all, SHMEM_INT2_MAX_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int4_max_to_all, SHMEM_INT4_MAX_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int8_max_to_all, SHMEM_INT8_MAX_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real4_max_to_all, SHMEM_REAL4_MAX_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real8_max_to_all, SHMEM_REAL8_MAX_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real16_max_to_all, SHMEM_REAL16_MAX_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int2_min_to_all, SHMEM_INT2_MIN_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int4_min_to_all, SHMEM_INT4_MIN_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int8_min_to_all, SHMEM_INT8_MIN_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real4_min_to_all, SHMEM_REAL4_MIN_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real8_min_to_all, SHMEM_REAL8_MIN_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real16_min_to_all, SHMEM_REAL16_MIN_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int2_sum_to_all, SHMEM_INT2_SUM_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int4_sum_to_all, SHMEM_INT4_SUM_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int8_sum_to_all, SHMEM_INT8_SUM_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_comp4_sum_to_all, SHMEM_COMP4_SUM_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_comp8_sum_to_all, SHMEM_COMP8_SUM_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real4_sum_to_all, SHMEM_REAL4_SUM_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real8_sum_to_all, SHMEM_REAL8_SUM_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real16_sum_to_all, SHMEM_REAL16_SUM_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int2_prod_to_all, SHMEM_INT2_PROD_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int4_prod_to_all, SHMEM_INT4_PROD_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_int8_prod_to_all, SHMEM_INT8_PROD_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_comp4_prod_to_all, SHMEM_COMP4_PROD_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_comp8_prod_to_all, SHMEM_COMP8_PROD_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real4_prod_to_all, SHMEM_REAL4_PROD_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real8_prod_to_all, SHMEM_REAL8_PROD_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_real16_prod_to_all, SHMEM_REAL16_PROD_TO_ALL, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nreduce, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T *pWrk, FORTRAN_POINTER_T pSync));
PN (void, shmem_collect4, SHMEM_COLLECT4, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_collect8, SHMEM_COLLECT8, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_collect32, SHMEM_COLLECT32, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_collect64, SHMEM_COLLECT64, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_fcollect4, SHMEM_FCOLLECT4, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_fcollect8, SHMEM_FCOLLECT8, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_fcollect32, SHMEM_FCOLLECT32, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_fcollect64, SHMEM_FCOLLECT64, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_broadcast4, SHMEM_BROADCAST4, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_root, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_broadcast8, SHMEM_BROADCAST8, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_root, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_broadcast32, SHMEM_BROADCAST32, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_root, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_broadcast64, SHMEM_BROADCAST64, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_root, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_alltoall32, SHMEM_ALLTOALL32, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_alltoall64, SHMEM_ALLTOALL64, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_alltoalls32, SHMEM_ALLTOALLS32, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *dst, MPI_Fint *sst, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_alltoalls64, SHMEM_ALLTOALLS64, (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *dst, MPI_Fint *sst, MPI_Fint *nlong, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync));
PN (void, shmem_set_lock, SHMEM_SET_LOCK, (FORTRAN_POINTER_T lock));
PN (void, shmem_clear_lock, SHMEM_CLEAR_LOCK, (FORTRAN_POINTER_T lock));
PN (MPI_Fint, shmem_test_lock, SHMEM_TEST_LOCK, (FORTRAN_POINTER_T lock));
PN (void, shmem_set_cache_inv, SHMEM_SET_CACHE_INV, (void));
PN (void, shmem_set_cache_line_inv, SHMEM_SET_CACHE_LINE_INV, (FORTRAN_POINTER_T target));
PN (void, shmem_clear_cache_inv, SHMEM_CLEAR_CACHE_INV, (void));
PN (void, shmem_clear_cache_line_inv, SHMEM_CLEAR_CACHE_LINE_INV, (FORTRAN_POINTER_T target));
PN (void, shmem_udcflush, SHMEM_UDCFLUSH, (void));
PN (void, shmem_udcflush_line, SHMEM_UDCFLUSH_LINE, (FORTRAN_POINTER_T target));
PN (void, shmem_fence, SHMEM_FENCE, (void));
PN (void, shmem_quiet, SHMEM_QUIET, (void));

END_C_DECLS
#endif /*OSHMEM_F77_PROTOTYPES_SHMEM_H*/
