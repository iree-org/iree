/*
 * Copyright (c) 2014      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2014      Intel, Inc. All rights reserved
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef PSHMEM_SHMEMX_H
#define PSHMEM_SHMEMX_H

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

/*
 * Symmetric heap routines
 */
OSHMEM_DECLSPEC  void* pshmemx_malloc_with_hint(size_t size, long hint);


/*
 * Legacy API
 * old init/destruct functions - not in the open shmem spec but still supported
 */
OSHMEM_DECLSPEC  void pstart_pes(int npes);

OSHMEM_DECLSPEC  int p_num_pes(void);
OSHMEM_DECLSPEC  int p_my_pe(void);

OSHMEM_DECLSPEC  void* pshmalloc(size_t size);
OSHMEM_DECLSPEC  void* pshmemalign(size_t align, size_t size);
OSHMEM_DECLSPEC  void* pshrealloc(void *ptr, size_t size);
OSHMEM_DECLSPEC  void pshfree(void* ptr);

OSHMEM_DECLSPEC  void pshmem_char_put(char *target, const char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void pshmem_char_get(char *target, const char *source, size_t len, int pe);


/*
 * All OpenSHMEM extension APIs that are not part of this specification must be defined in the shmemx.h include
 * file. These extensions shall use the shmemx_ prefix for all routine, variable, and constant names.
 */

/*
 * Elemental put routines
 */
OSHMEM_DECLSPEC  void pshmemx_int16_p(int16_t* addr, int16_t value, int pe);
OSHMEM_DECLSPEC  void pshmemx_int32_p(int32_t* addr, int32_t value, int pe);
OSHMEM_DECLSPEC  void pshmemx_int64_p(int64_t* addr, int64_t value, int pe);

/*
 * Block data put routines
 */

/*
 * Strided put routines
 */

/*
 * Elemental get routines
 */
OSHMEM_DECLSPEC  int16_t pshmemx_int16_g(const int16_t* addr, int pe);
OSHMEM_DECLSPEC  int32_t pshmemx_int32_g(const int32_t* addr, int pe);
OSHMEM_DECLSPEC  int64_t pshmemx_int64_g(const int64_t* addr, int pe);

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
OSHMEM_DECLSPEC int32_t pshmemx_int32_swap(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t pshmemx_int64_swap(int64_t *target, int64_t value, int pe);

/*Atomic set */
OSHMEM_DECLSPEC void pshmemx_int32_set(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void pshmemx_int64_set(int64_t *target, int64_t value, int pe);

/* Atomic conditional swap */
OSHMEM_DECLSPEC int32_t pshmemx_int32_cswap(int32_t *target, int32_t cond, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t pshmemx_int64_cswap(int64_t *target, int64_t cond, int64_t value, int pe);

/* Atomic Fetch&Add */
OSHMEM_DECLSPEC int32_t pshmemx_int32_fadd(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t pshmemx_int64_fadd(int64_t *target, int64_t value, int pe);

/* Atomic Fetch&And */
OSHMEM_DECLSPEC int32_t pshmemx_int32_atomic_fetch_and(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t pshmemx_int64_atomic_fetch_and(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t pshmemx_uint32_atomic_fetch_and(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t pshmemx_uint64_atomic_fetch_and(uint64_t *target, uint64_t value, int pe);

/* Atomic Fetch&Or */
OSHMEM_DECLSPEC int32_t pshmemx_int32_atomic_fetch_or(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t pshmemx_int64_atomic_fetch_or(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t pshmemx_uint32_atomic_fetch_or(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t pshmemx_uint64_atomic_fetch_or(uint64_t *target, uint64_t value, int pe);

/* Atomic Fetch&Xor */
OSHMEM_DECLSPEC int32_t pshmemx_int32_atomic_fetch_xor(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t pshmemx_int64_atomic_fetch_xor(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t pshmemx_uint32_atomic_fetch_xor(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t pshmemx_uint64_atomic_fetch_xor(uint64_t *target, uint64_t value, int pe);

/* Atomic Fetch */
OSHMEM_DECLSPEC int32_t pshmemx_int32_fetch(const int32_t *target, int pe);
OSHMEM_DECLSPEC int64_t pshmemx_int64_fetch(const int64_t *target, int pe);

/* Atomic Fetch&Inc */
OSHMEM_DECLSPEC int32_t pshmemx_int32_finc(int32_t *target, int pe);
OSHMEM_DECLSPEC int64_t pshmemx_int64_finc(int64_t *target, int pe);

/* Atomic Add */
OSHMEM_DECLSPEC void pshmemx_int32_add(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void pshmemx_int64_add(int64_t *target, int64_t value, int pe);

/* Atomic And */
OSHMEM_DECLSPEC void pshmemx_int32_atomic_and(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void pshmemx_int64_atomic_and(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void pshmemx_uint32_atomic_and(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void pshmemx_uint64_atomic_and(uint64_t *target, uint64_t value, int pe);

/* Atomic Or */
OSHMEM_DECLSPEC void pshmemx_int32_atomic_or(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void pshmemx_int64_atomic_or(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void pshmemx_uint32_atomic_or(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void pshmemx_uint64_atomic_or(uint64_t *target, uint64_t value, int pe);

/* Atomic Xor */
OSHMEM_DECLSPEC void pshmemx_int32_atomic_xor(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void pshmemx_int64_atomic_xor(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void pshmemx_uint32_atomic_xor(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void pshmemx_uint64_atomic_xor(uint64_t *target, uint64_t value, int pe);

/* Atomic Inc */
OSHMEM_DECLSPEC void pshmemx_int32_inc(int32_t *target, int pe);
OSHMEM_DECLSPEC void pshmemx_int64_inc(int64_t *target, int pe);

/*
 * P2P sync routines
 */
OSHMEM_DECLSPEC  void pshmemx_int32_wait(int32_t *addr, int32_t value);
OSHMEM_DECLSPEC  void pshmemx_int64_wait(int64_t *addr, int64_t value);

OSHMEM_DECLSPEC  void pshmemx_int32_wait_until(int32_t *addr, int cmp, int32_t value);
OSHMEM_DECLSPEC  void pshmemx_int64_wait_until(int64_t *addr, int cmp, int64_t value);

/*
 * Reduction routines
 */
OSHMEM_DECLSPEC void pshmemx_int16_and_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int32_and_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int64_and_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void pshmemx_int16_or_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int32_or_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int64_or_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void pshmemx_int16_xor_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int32_xor_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int64_xor_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void pshmemx_int16_max_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int32_max_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int64_max_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void pshmemx_int16_min_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int32_min_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int64_min_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void pshmemx_int16_sum_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int32_sum_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int64_sum_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

OSHMEM_DECLSPEC void pshmemx_int16_prod_to_all(int16_t *target, const int16_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int16_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int32_prod_to_all(int32_t *target, const int32_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int32_t *pWrk, long *pSync);
OSHMEM_DECLSPEC void pshmemx_int64_prod_to_all(int64_t *target, const int64_t *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int64_t *pWrk, long *pSync);

/*
 * Backward compatibility section
 */
#define pshmem_int32_swap            pshmemx_int32_swap
#define pshmem_int64_swap            pshmemx_int64_swap

#define pshmem_int32_set             pshmemx_int32_set
#define pshmem_int64_set             pshmemx_int64_set

#define pshmem_int32_cswap           pshmemx_int32_cswap
#define pshmem_int64_cswap           pshmemx_int64_cswap

#define pshmem_int32_fadd            pshmemx_int32_fadd
#define pshmem_int64_fadd            pshmemx_int64_fadd

#define pshmem_int32_fetch           pshmemx_int32_fetch
#define pshmem_int64_fetch           pshmemx_int64_fetch

#define pshmem_int32_finc            pshmemx_int32_finc
#define pshmem_int64_finc            pshmemx_int64_finc

#define pshmem_int32_add             pshmemx_int32_add
#define pshmem_int64_add             pshmemx_int64_add

#define pshmem_int32_inc             pshmemx_int32_inc
#define pshmem_int64_inc             pshmemx_int64_inc

#define pshmem_int32_wait            pshmemx_int32_wait
#define pshmem_int64_wait            pshmemx_int64_wait

#define pshmem_int16_and_to_all      pshmemx_int16_and_to_all
#define pshmem_int32_and_to_all      pshmemx_int32_and_to_all
#define pshmem_int64_and_to_all      pshmemx_int64_and_to_all

#define pshmem_int16_or_to_all       pshmemx_int16_or_to_all
#define pshmem_int32_or_to_all       pshmemx_int32_or_to_all
#define pshmem_int64_or_to_all       pshmemx_int64_or_to_all

#define pshmem_int16_xor_to_all      pshmemx_int16_xor_to_all
#define pshmem_int32_xor_to_all      pshmemx_int32_xor_to_all
#define pshmem_int64_xor_to_all      pshmemx_int64_xor_to_all

#define pshmem_int16_max_to_all      pshmemx_int16_max_to_all
#define pshmem_int32_max_to_all      pshmemx_int32_max_to_all
#define pshmem_int64_max_to_all      pshmemx_int64_max_to_all

#define pshmem_int16_min_to_all      pshmemx_int16_min_to_all
#define pshmem_int32_min_to_all      pshmemx_int32_min_to_all
#define pshmem_int64_min_to_all      pshmemx_int64_min_to_all

#define pshmem_int16_sum_to_all      pshmemx_int16_sum_to_all
#define pshmem_int32_sum_to_all      pshmemx_int32_sum_to_all
#define pshmem_int64_sum_to_all      pshmemx_int64_sum_to_all

#define pshmem_int16_prod_to_all     pshmemx_int16_prod_to_all
#define pshmem_int32_prod_to_all     pshmemx_int32_prod_to_all
#define pshmem_int64_prod_to_all     pshmemx_int64_prod_to_all

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif /* PSHMEM_SHMEMX_H */
