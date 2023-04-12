/* oshmem/include/shmemx.h. This file contains vendor extension functions  */
/*
 * Copyright (c) 2014-2015 Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OSHMEM_SHMEMX_H
#define OSHMEM_SHMEMX_H

#include <shmem.h>

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

enum {
    SHMEM_HINT_NONE           = 0,
    SHMEM_HINT_LOW_LAT_MEM    = 1 << 0,
    SHMEM_HINT_HIGH_BW_MEM    = 1 << 1,
    SHMEM_HINT_NEAR_NIC_MEM   = 1 << 2,
    SHMEM_HINT_DEVICE_GPU_MEM = 1 << 3,
    SHMEM_HINT_DEVICE_NIC_MEM = 1 << 4,

    SHMEM_HINT_PSYNC          = 1 << 16,
    SHMEM_HINT_PWORK          = 1 << 17,
    SHMEM_HINT_ATOMICS        = 1 << 18
};

/*
 * All OpenSHMEM extension APIs that are not part of this specification must be defined in the shmemx.h include
 * file. These extensions shall use the shmemx_ prefix for all routine, variable, and constant names.
 */

/*
 * Symmetric heap routines
 */
OSHMEM_DECLSPEC  void* shmemx_malloc_with_hint(size_t size, long hint);

/*
 * Elemental put routines
 */
OSHMEM_DECLSPEC  void shmemx_int16_p(int16_t* addr, int16_t value, int pe);
OSHMEM_DECLSPEC  void shmemx_int32_p(int32_t* addr, int32_t value, int pe);
OSHMEM_DECLSPEC  void shmemx_int64_p(int64_t* addr, int64_t value, int pe);

/*
 * Elemental put routines
 */

/*
 * Block data put routines
 */

/*
 * Strided put routines
 */

/*
 * Elemental get routines
 */
OSHMEM_DECLSPEC  int16_t shmemx_int16_g(const int16_t* addr, int pe);
OSHMEM_DECLSPEC  int32_t shmemx_int32_g(const int32_t* addr, int pe);
OSHMEM_DECLSPEC  int64_t shmemx_int64_g(const int64_t* addr, int pe);

/*
 * Block data get routines
 */

/*
 * Strided get routines
 */

/*
 * Atomic operations
 */
/* Atomic swap */
OSHMEM_DECLSPEC int32_t shmemx_int32_swap(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmemx_int64_swap(int64_t *target, int64_t value, int pe);

/* Atomic set */
OSHMEM_DECLSPEC void shmemx_int32_set(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void shmemx_int64_set(int64_t *target, int64_t value, int pe);

/* Atomic conditional swap */
OSHMEM_DECLSPEC int32_t shmemx_int32_cswap(int32_t *target, int32_t cond, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmemx_int64_cswap(int64_t *target, int64_t cond, int64_t value, int pe);

/* Atomic Fetch&Add */
OSHMEM_DECLSPEC int32_t shmemx_int32_fadd(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmemx_int64_fadd(int64_t *target, int64_t value, int pe);

/* Atomic Fetch&And */
OSHMEM_DECLSPEC int32_t shmemx_int32_atomic_fetch_and(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmemx_int64_atomic_fetch_and(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t shmemx_uint32_atomic_fetch_and(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t shmemx_uint64_atomic_fetch_and(uint64_t *target, uint64_t value, int pe);

/* Atomic Fetch&Or */
OSHMEM_DECLSPEC int32_t shmemx_int32_atomic_fetch_or(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmemx_int64_atomic_fetch_or(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t shmemx_uint32_atomic_fetch_or(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t shmemx_uint64_atomic_fetch_or(uint64_t *target, uint64_t value, int pe);

/* Atomic Fetch&Xor */
OSHMEM_DECLSPEC int32_t shmemx_int32_atomic_fetch_xor(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmemx_int64_atomic_fetch_xor(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t shmemx_uint32_atomic_fetch_xor(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t shmemx_uint64_atomic_fetch_xor(uint64_t *target, uint64_t value, int pe);

/* Atomic Fetch */
OSHMEM_DECLSPEC int32_t shmemx_int32_fetch(const int32_t *target, int pe);
OSHMEM_DECLSPEC int64_t shmemx_int64_fetch(const int64_t *target, int pe);

/* Atomic Fetch&Inc */
OSHMEM_DECLSPEC int32_t shmemx_int32_finc(int32_t *target, int pe);
OSHMEM_DECLSPEC int64_t shmemx_int64_finc(int64_t *target, int pe);

/* Atomic Add */
OSHMEM_DECLSPEC void shmemx_int32_add(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void shmemx_int64_add(int64_t *target, int64_t value, int pe);

/* Atomic And */
OSHMEM_DECLSPEC void shmemx_int32_atomic_and(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void shmemx_int64_atomic_and(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void shmemx_uint32_atomic_and(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void shmemx_uint64_atomic_and(uint64_t *target, uint64_t value, int pe);

/* Atomic Or */
OSHMEM_DECLSPEC void shmemx_int32_atomic_or(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void shmemx_int64_atomic_or(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void shmemx_uint32_atomic_or(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void shmemx_uint64_atomic_or(uint64_t *target, uint64_t value, int pe);

/* Atomic Xor */
OSHMEM_DECLSPEC void shmemx_int32_atomic_xor(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void shmemx_int64_atomic_xor(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void shmemx_uint32_atomic_xor(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void shmemx_uint64_atomic_xor(uint64_t *target, uint64_t value, int pe);

/* Atomic Inc */
OSHMEM_DECLSPEC void shmemx_int32_inc(int32_t *target, int pe);
OSHMEM_DECLSPEC void shmemx_int64_inc(int64_t *target, int pe);

/*
 * P2P sync routines
 */
OSHMEM_DECLSPEC  void shmemx_int32_wait(int32_t *addr, int32_t value);
OSHMEM_DECLSPEC  void shmemx_int64_wait(int64_t *addr, int64_t value);

OSHMEM_DECLSPEC  void shmemx_int32_wait_until(int32_t *addr, int cmp, int32_t value);
OSHMEM_DECLSPEC  void shmemx_int64_wait_until(int64_t *addr, int cmp, int64_t value);

/*
 * Reduction routines
 */
OSHMEM_DECLSPEC void shmemx_int16_and_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int32_and_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int64_and_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmemx_int16_or_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int32_or_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int64_or_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmemx_int16_xor_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int32_xor_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int64_xor_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmemx_int16_max_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int32_max_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int64_max_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmemx_int16_min_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int32_min_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int64_min_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmemx_int16_sum_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int32_sum_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int64_sum_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmemx_int16_prod_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int32_prod_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmemx_int64_prod_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

/* shmemx_alltoall_global_nb is a nonblocking collective routine, where each PE
 * exchanges “size” bytes of data with all other PEs in the OpenSHMEM job.

 *  @param dest        A symmetric data object that is large enough to receive
 *                     “size” bytes of data from each PE in the OpenSHMEM job.
 *  @param source      A symmetric data object that contains “size” bytes of data
 *                     for each PE in the OpenSHMEM job.
 *  @param size        The number of bytes to be sent to each PE in the job.
 *  @param counter     A symmetric data object to be atomically incremented after
 *                     the target buffer is updated.
 *
 *  @return            OSHMEM_SUCCESS or failure status.
 */
OSHMEM_DECLSPEC void shmemx_alltoall_global_nb(void *dest, const void *source, size_t size, long *counter);

/*
 * Backward compatibility section
 */
#define shmem_int32_swap            shmemx_int32_swap
#define shmem_int64_swap            shmemx_int64_swap

#define shmem_int32_set             shmemx_int32_set
#define shmem_int64_set             shmemx_int64_set

#define shmem_int32_cswap           shmemx_int32_cswap
#define shmem_int64_cswap           shmemx_int64_cswap

#define shmem_int32_fadd            shmemx_int32_fadd
#define shmem_int64_fadd            shmemx_int64_fadd

#define shmem_int32_fetch           shmemx_int32_fetch
#define shmem_int64_fetch           shmemx_int64_fetch

#define shmem_int32_finc            shmemx_int32_finc
#define shmem_int64_finc            shmemx_int64_finc

#define shmem_int32_add             shmemx_int32_add
#define shmem_int64_add             shmemx_int64_add
#define shmem_int32_inc             shmemx_int32_inc
#define shmem_int64_inc             shmemx_int64_inc

#define shmem_int32_wait            shmemx_int32_wait
#define shmem_int64_wait            shmemx_int64_wait

#define shmem_int16_and_to_all      shmemx_int16_and_to_all
#define shmem_int32_and_to_all      shmemx_int32_and_to_all
#define shmem_int64_and_to_all      shmemx_int64_and_to_all

#define shmem_int16_or_to_all       shmemx_int16_or_to_all
#define shmem_int32_or_to_all       shmemx_int32_or_to_all
#define shmem_int64_or_to_all       shmemx_int64_or_to_all

#define shmem_int16_xor_to_all      shmemx_int16_xor_to_all
#define shmem_int32_xor_to_all      shmemx_int32_xor_to_all
#define shmem_int64_xor_to_all      shmemx_int64_xor_to_all

#define shmem_int16_max_to_all      shmemx_int16_max_to_all
#define shmem_int32_max_to_all      shmemx_int32_max_to_all
#define shmem_int64_max_to_all      shmemx_int64_max_to_all

#define shmem_int16_min_to_all      shmemx_int16_min_to_all
#define shmem_int32_min_to_all      shmemx_int32_min_to_all
#define shmem_int64_min_to_all      shmemx_int64_min_to_all

#define shmem_int16_sum_to_all      shmemx_int16_sum_to_all
#define shmem_int32_sum_to_all      shmemx_int32_sum_to_all
#define shmem_int64_sum_to_all      shmemx_int64_sum_to_all

#define shmem_int16_prod_to_all     shmemx_int16_prod_to_all
#define shmem_int32_prod_to_all     shmemx_int32_prod_to_all
#define shmem_int64_prod_to_all     shmemx_int64_prod_to_all

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif /* OSHMEM_SHMEMX_H */
