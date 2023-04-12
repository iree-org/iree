/*
 * Copyright (c) 2013-2017 Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OSHMEM_C_PROFILE_DEFINES_H
#define OSHMEM_C_PROFILE_DEFINES_H
/*
 * This file is included in the top directory only if
 * profiling is required. Once profiling is required,
 * this file will replace all shmem_* symbols with
 * pshmem_* symbols
 */

/*
 * Initialization routines
 */
#define shmem_init                   pshmem_init
#define shmem_init_thread            pshmem_init_thread
#define start_pes                    pstart_pes /* shmem-compat.h */

/*
 * Finalization routines
 */
#define shmem_finalize               pshmem_finalize
#define shmem_global_exit            pshmem_global_exit

/*
 * Query routines
 */
#define shmem_n_pes                  pshmem_n_pes
#define shmem_query_thread           pshmem_query_thread
#define shmem_my_pe                  pshmem_my_pe
#define _num_pes                     p_num_pes /* shmem-compat.h */
#define _my_pe                       p_my_pe /* shmem-compat.h */

/*
 * Accessability routines
 */
#define shmem_pe_accessible          pshmem_pe_accessible
#define shmem_addr_accessible        pshmem_addr_accessible

/*
 * Symmetric heap routines
 */
#define shmem_malloc                 pshmem_malloc
#define shmem_calloc                 pshmem_calloc
#define shmem_align                  pshmem_align
#define shmem_realloc                pshmem_realloc
#define shmem_free                   pshmem_free
#define shmalloc                     pshmalloc /* shmem-compat.h */
#define shmemalign                   pshmemalign /* shmem-compat.h */
#define shrealloc                    pshrealloc /* shmem-compat.h */
#define shfree                       pshfree /* shmem-compat.h */

#define shmemx_malloc_with_hint      pshmemx_malloc_with_hint

/*
 * Remote pointer operations
 */
#define shmem_ptr                    pshmem_ptr

/*
 * Communication context operations
 */
#define shmem_ctx_create             pshmem_ctx_create
#define shmem_ctx_destroy            pshmem_ctx_destroy

/*
 * Elemental put routines
 */
#define shmem_ctx_char_p             pshmem_ctx_char_p
#define shmem_ctx_short_p            pshmem_ctx_short_p
#define shmem_ctx_int_p              pshmem_ctx_int_p
#define shmem_ctx_long_p             pshmem_ctx_long_p
#define shmem_ctx_float_p            pshmem_ctx_float_p
#define shmem_ctx_double_p           pshmem_ctx_double_p
#define shmem_ctx_longlong_p         pshmem_ctx_longlong_p
#define shmem_ctx_schar_p            pshmem_ctx_schar_p
#define shmem_ctx_uchar_p            pshmem_ctx_uchar_p
#define shmem_ctx_ushort_p           pshmem_ctx_ushort_p
#define shmem_ctx_uint_p             pshmem_ctx_uint_p
#define shmem_ctx_ulong_p            pshmem_ctx_ulong_p
#define shmem_ctx_ulonglong_p        pshmem_ctx_ulonglong_p
#define shmem_ctx_longdouble_p       pshmem_ctx_longdouble_p
#define shmem_ctx_int8_p             pshmem_ctx_int8_p
#define shmem_ctx_int16_p            pshmem_ctx_int16_p
#define shmem_ctx_int32_p            pshmem_ctx_int32_p
#define shmem_ctx_int64_p            pshmem_ctx_int64_p
#define shmem_ctx_uint8_p            pshmem_ctx_uint8_p
#define shmem_ctx_uint16_p           pshmem_ctx_uint16_p
#define shmem_ctx_uint32_p           pshmem_ctx_uint32_p
#define shmem_ctx_uint64_p           pshmem_ctx_uint64_p
#define shmem_ctx_size_p             pshmem_ctx_size_p
#define shmem_ctx_ptrdiff_p          pshmem_ctx_ptrdiff_p

#define shmem_char_p                 pshmem_char_p
#define shmem_short_p                pshmem_short_p
#define shmem_int_p                  pshmem_int_p
#define shmem_long_p                 pshmem_long_p
#define shmem_float_p                pshmem_float_p
#define shmem_double_p               pshmem_double_p
#define shmem_longlong_p             pshmem_longlong_p
#define shmem_schar_p                pshmem_schar_p
#define shmem_uchar_p                pshmem_uchar_p
#define shmem_ushort_p               pshmem_ushort_p
#define shmem_uint_p                 pshmem_uint_p
#define shmem_ulong_p                pshmem_ulong_p
#define shmem_ulonglong_p            pshmem_ulonglong_p
#define shmem_longdouble_p           pshmem_longdouble_p
#define shmem_int8_p                 pshmem_int8_p
#define shmem_int16_p                pshmem_int16_p
#define shmem_int32_p                pshmem_int32_p
#define shmem_int64_p                pshmem_int64_p
#define shmem_uint8_p                pshmem_uint8_p
#define shmem_uint16_p               pshmem_uint16_p
#define shmem_uint32_p               pshmem_uint32_p
#define shmem_uint64_p               pshmem_uint64_p
#define shmem_size_p                 pshmem_size_p
#define shmem_ptrdiff_p              pshmem_ptrdiff_p

#define shmemx_int16_p               pshmemx_int16_p
#define shmemx_int32_p               pshmemx_int32_p
#define shmemx_int64_p               pshmemx_int64_p

/*
 * Block data put routines
 */
#define shmem_ctx_char_put           pshmem_ctx_char_put
#define shmem_ctx_short_put          pshmem_ctx_short_put
#define shmem_ctx_int_put            pshmem_ctx_int_put
#define shmem_ctx_long_put           pshmem_ctx_long_put
#define shmem_ctx_float_put          pshmem_ctx_float_put
#define shmem_ctx_double_put         pshmem_ctx_double_put
#define shmem_ctx_longlong_put       pshmem_ctx_longlong_put
#define shmem_ctx_schar_put          pshmem_ctx_schar_put
#define shmem_ctx_uchar_put          pshmem_ctx_uchar_put
#define shmem_ctx_ushort_put         pshmem_ctx_ushort_put
#define shmem_ctx_uint_put           pshmem_ctx_uint_put
#define shmem_ctx_ulong_put          pshmem_ctx_ulong_put
#define shmem_ctx_ulonglong_put      pshmem_ctx_ulonglong_put
#define shmem_ctx_longdouble_put     pshmem_ctx_longdouble_put
#define shmem_ctx_int8_put           pshmem_ctx_int8_put
#define shmem_ctx_int16_put          pshmem_ctx_int16_put
#define shmem_ctx_int32_put          pshmem_ctx_int32_put
#define shmem_ctx_int64_put          pshmem_ctx_int64_put
#define shmem_ctx_uint8_put          pshmem_ctx_uint8_put
#define shmem_ctx_uint16_put         pshmem_ctx_uint16_put
#define shmem_ctx_uint32_put         pshmem_ctx_uint32_put
#define shmem_ctx_uint64_put         pshmem_ctx_uint64_put
#define shmem_ctx_size_put           pshmem_ctx_size_put
#define shmem_ctx_ptrdiff_put        pshmem_ctx_ptrdiff_put

#define shmem_char_put               pshmem_char_put /* shmem-compat.h */
#define shmem_short_put              pshmem_short_put
#define shmem_int_put                pshmem_int_put
#define shmem_long_put               pshmem_long_put
#define shmem_float_put              pshmem_float_put
#define shmem_double_put             pshmem_double_put
#define shmem_longlong_put           pshmem_longlong_put
#define shmem_schar_put              pshmem_schar_put
#define shmem_uchar_put              pshmem_uchar_put
#define shmem_ushort_put             pshmem_ushort_put
#define shmem_uint_put               pshmem_uint_put
#define shmem_ulong_put              pshmem_ulong_put
#define shmem_ulonglong_put          pshmem_ulonglong_put
#define shmem_longdouble_put         pshmem_longdouble_put
#define shmem_int8_put               pshmem_int8_put
#define shmem_int16_put              pshmem_int16_put
#define shmem_int32_put              pshmem_int32_put
#define shmem_int64_put              pshmem_int64_put
#define shmem_uint8_put              pshmem_uint8_put
#define shmem_uint16_put             pshmem_uint16_put
#define shmem_uint32_put             pshmem_uint32_put
#define shmem_uint64_put             pshmem_uint64_put
#define shmem_size_put               pshmem_size_put
#define shmem_ptrdiff_put            pshmem_ptrdiff_put

#define shmem_ctx_put8               pshmem_ctx_put8
#define shmem_ctx_put16              pshmem_ctx_put16
#define shmem_ctx_put32              pshmem_ctx_put32
#define shmem_ctx_put64              pshmem_ctx_put64
#define shmem_ctx_put128             pshmem_ctx_put128
#define shmem_ctx_putmem             pshmem_ctx_putmem

#define shmem_put8                   pshmem_put8
#define shmem_put16                  pshmem_put16
#define shmem_put32                  pshmem_put32
#define shmem_put64                  pshmem_put64
#define shmem_put128                 pshmem_put128
#define shmem_putmem                 pshmem_putmem

/*
 * Strided put routines
 */
#define shmem_ctx_char_iput           pshmem_ctx_char_iput
#define shmem_ctx_short_iput          pshmem_ctx_short_iput
#define shmem_ctx_int_iput            pshmem_ctx_int_iput
#define shmem_ctx_long_iput           pshmem_ctx_long_iput
#define shmem_ctx_float_iput          pshmem_ctx_float_iput
#define shmem_ctx_double_iput         pshmem_ctx_double_iput
#define shmem_ctx_longlong_iput       pshmem_ctx_longlong_iput
#define shmem_ctx_schar_iput          pshmem_ctx_schar_iput
#define shmem_ctx_uchar_iput          pshmem_ctx_uchar_iput
#define shmem_ctx_ushort_iput         pshmem_ctx_ushort_iput
#define shmem_ctx_uint_iput           pshmem_ctx_uint_iput
#define shmem_ctx_ulong_iput          pshmem_ctx_ulong_iput
#define shmem_ctx_ulonglong_iput      pshmem_ctx_ulonglong_iput
#define shmem_ctx_longdouble_iput     pshmem_ctx_longdouble_iput
#define shmem_ctx_int8_iput           pshmem_ctx_int8_iput
#define shmem_ctx_int16_iput          pshmem_ctx_int16_iput
#define shmem_ctx_int32_iput          pshmem_ctx_int32_iput
#define shmem_ctx_int64_iput          pshmem_ctx_int64_iput
#define shmem_ctx_uint8_iput          pshmem_ctx_uint8_iput
#define shmem_ctx_uint16_iput         pshmem_ctx_uint16_iput
#define shmem_ctx_uint32_iput         pshmem_ctx_uint32_iput
#define shmem_ctx_uint64_iput         pshmem_ctx_uint64_iput
#define shmem_ctx_size_iput           pshmem_ctx_size_iput
#define shmem_ctx_ptrdiff_iput        pshmem_ctx_ptrdiff_iput

#define shmem_char_iput               pshmem_char_iput
#define shmem_short_iput              pshmem_short_iput
#define shmem_int_iput                pshmem_int_iput
#define shmem_long_iput               pshmem_long_iput
#define shmem_float_iput              pshmem_float_iput
#define shmem_double_iput             pshmem_double_iput
#define shmem_longlong_iput           pshmem_longlong_iput
#define shmem_schar_iput              pshmem_schar_iput
#define shmem_uchar_iput              pshmem_uchar_iput
#define shmem_ushort_iput             pshmem_ushort_iput
#define shmem_uint_iput               pshmem_uint_iput
#define shmem_ulong_iput              pshmem_ulong_iput
#define shmem_ulonglong_iput          pshmem_ulonglong_iput
#define shmem_longdouble_iput         pshmem_longdouble_iput
#define shmem_int8_iput               pshmem_int8_iput
#define shmem_int16_iput              pshmem_int16_iput
#define shmem_int32_iput              pshmem_int32_iput
#define shmem_int64_iput              pshmem_int64_iput
#define shmem_uint8_iput              pshmem_uint8_iput
#define shmem_uint16_iput             pshmem_uint16_iput
#define shmem_uint32_iput             pshmem_uint32_iput
#define shmem_uint64_iput             pshmem_uint64_iput
#define shmem_size_iput               pshmem_size_iput
#define shmem_ptrdiff_iput            pshmem_ptrdiff_iput

#define shmem_ctx_iput8              pshmem_ctx_iput8
#define shmem_ctx_iput16             pshmem_ctx_iput16
#define shmem_ctx_iput32             pshmem_ctx_iput32
#define shmem_ctx_iput64             pshmem_ctx_iput64
#define shmem_ctx_iput128            pshmem_ctx_iput128

#define shmem_iput8                  pshmem_iput8
#define shmem_iput16                 pshmem_iput16
#define shmem_iput32                 pshmem_iput32
#define shmem_iput64                 pshmem_iput64
#define shmem_iput128                pshmem_iput128

/*
 * Non-block data put routines
 */
#define shmem_ctx_char_put_nbi           pshmem_ctx_char_put_nbi
#define shmem_ctx_short_put_nbi          pshmem_ctx_short_put_nbi
#define shmem_ctx_int_put_nbi            pshmem_ctx_int_put_nbi
#define shmem_ctx_long_put_nbi           pshmem_ctx_long_put_nbi
#define shmem_ctx_float_put_nbi          pshmem_ctx_float_put_nbi
#define shmem_ctx_double_put_nbi         pshmem_ctx_double_put_nbi
#define shmem_ctx_longlong_put_nbi       pshmem_ctx_longlong_put_nbi
#define shmem_ctx_schar_put_nbi          pshmem_ctx_schar_put_nbi
#define shmem_ctx_uchar_put_nbi          pshmem_ctx_uchar_put_nbi
#define shmem_ctx_ushort_put_nbi         pshmem_ctx_ushort_put_nbi
#define shmem_ctx_uint_put_nbi           pshmem_ctx_uint_put_nbi
#define shmem_ctx_ulong_put_nbi          pshmem_ctx_ulong_put_nbi
#define shmem_ctx_ulonglong_put_nbi      pshmem_ctx_ulonglong_put_nbi
#define shmem_ctx_longdouble_put_nbi     pshmem_ctx_longdouble_put_nbi
#define shmem_ctx_int8_put_nbi           pshmem_ctx_int8_put_nbi
#define shmem_ctx_int16_put_nbi          pshmem_ctx_int16_put_nbi
#define shmem_ctx_int32_put_nbi          pshmem_ctx_int32_put_nbi
#define shmem_ctx_int64_put_nbi          pshmem_ctx_int64_put_nbi
#define shmem_ctx_uint8_put_nbi          pshmem_ctx_uint8_put_nbi
#define shmem_ctx_uint16_put_nbi         pshmem_ctx_uint16_put_nbi
#define shmem_ctx_uint32_put_nbi         pshmem_ctx_uint32_put_nbi
#define shmem_ctx_uint64_put_nbi         pshmem_ctx_uint64_put_nbi
#define shmem_ctx_size_put_nbi           pshmem_ctx_size_put_nbi
#define shmem_ctx_ptrdiff_put_nbi        pshmem_ctx_ptrdiff_put_nbi

#define shmem_char_put_nbi               pshmem_char_put_nbi
#define shmem_short_put_nbi              pshmem_short_put_nbi
#define shmem_int_put_nbi                pshmem_int_put_nbi
#define shmem_long_put_nbi               pshmem_long_put_nbi
#define shmem_float_put_nbi              pshmem_float_put_nbi
#define shmem_double_put_nbi             pshmem_double_put_nbi
#define shmem_longlong_put_nbi           pshmem_longlong_put_nbi
#define shmem_schar_put_nbi              pshmem_schar_put_nbi
#define shmem_uchar_put_nbi              pshmem_uchar_put_nbi
#define shmem_ushort_put_nbi             pshmem_ushort_put_nbi
#define shmem_uint_put_nbi               pshmem_uint_put_nbi
#define shmem_ulong_put_nbi              pshmem_ulong_put_nbi
#define shmem_ulonglong_put_nbi          pshmem_ulonglong_put_nbi
#define shmem_longdouble_put_nbi         pshmem_longdouble_put_nbi
#define shmem_int8_put_nbi               pshmem_int8_put_nbi
#define shmem_int16_put_nbi              pshmem_int16_put_nbi
#define shmem_int32_put_nbi              pshmem_int32_put_nbi
#define shmem_int64_put_nbi              pshmem_int64_put_nbi
#define shmem_uint8_put_nbi              pshmem_uint8_put_nbi
#define shmem_uint16_put_nbi             pshmem_uint16_put_nbi
#define shmem_uint32_put_nbi             pshmem_uint32_put_nbi
#define shmem_uint64_put_nbi             pshmem_uint64_put_nbi
#define shmem_size_put_nbi               pshmem_size_put_nbi
#define shmem_ptrdiff_put_nbi            pshmem_ptrdiff_put_nbi

#define shmem_ctx_put8_nbi           pshmem_ctx_put8_nbi
#define shmem_ctx_put16_nbi          pshmem_ctx_put16_nbi
#define shmem_ctx_put32_nbi          pshmem_ctx_put32_nbi
#define shmem_ctx_put64_nbi          pshmem_ctx_put64_nbi
#define shmem_ctx_put128_nbi         pshmem_ctx_put128_nbi
#define shmem_ctx_putmem_nbi         pshmem_ctx_putmem_nbi

#define shmem_put8_nbi               pshmem_put8_nbi
#define shmem_put16_nbi              pshmem_put16_nbi
#define shmem_put32_nbi              pshmem_put32_nbi
#define shmem_put64_nbi              pshmem_put64_nbi
#define shmem_put128_nbi             pshmem_put128_nbi
#define shmem_putmem_nbi             pshmem_putmem_nbi

/*
 * Elemental get routines
 */
#define shmem_ctx_char_g             pshmem_ctx_char_g
#define shmem_ctx_short_g            pshmem_ctx_short_g
#define shmem_ctx_int_g              pshmem_ctx_int_g
#define shmem_ctx_long_g             pshmem_ctx_long_g
#define shmem_ctx_float_g            pshmem_ctx_float_g
#define shmem_ctx_double_g           pshmem_ctx_double_g
#define shmem_ctx_longlong_g         pshmem_ctx_longlong_g
#define shmem_ctx_schar_g            pshmem_ctx_schar_g
#define shmem_ctx_uchar_g            pshmem_ctx_uchar_g
#define shmem_ctx_ushort_g           pshmem_ctx_ushort_g
#define shmem_ctx_uint_g             pshmem_ctx_uint_g
#define shmem_ctx_ulong_g            pshmem_ctx_ulong_g
#define shmem_ctx_ulonglong_g        pshmem_ctx_ulonglong_g
#define shmem_ctx_longdouble_g       pshmem_ctx_longdouble_g
#define shmem_ctx_int8_g             pshmem_ctx_int8_g
#define shmem_ctx_int16_g            pshmem_ctx_int16_g
#define shmem_ctx_int32_g            pshmem_ctx_int32_g
#define shmem_ctx_int64_g            pshmem_ctx_int64_g
#define shmem_ctx_uint8_g            pshmem_ctx_uint8_g
#define shmem_ctx_uint16_g           pshmem_ctx_uint16_g
#define shmem_ctx_uint32_g           pshmem_ctx_uint32_g
#define shmem_ctx_uint64_g           pshmem_ctx_uint64_g
#define shmem_ctx_size_g             pshmem_ctx_size_g
#define shmem_ctx_ptrdiff_g          pshmem_ctx_ptrdiff_g

#define shmem_char_g                 pshmem_char_g
#define shmem_short_g                pshmem_short_g
#define shmem_int_g                  pshmem_int_g
#define shmem_long_g                 pshmem_long_g
#define shmem_float_g                pshmem_float_g
#define shmem_double_g               pshmem_double_g
#define shmem_longlong_g             pshmem_longlong_g
#define shmem_schar_g                pshmem_schar_g
#define shmem_uchar_g                pshmem_uchar_g
#define shmem_ushort_g               pshmem_ushort_g
#define shmem_uint_g                 pshmem_uint_g
#define shmem_ulong_g                pshmem_ulong_g
#define shmem_ulonglong_g            pshmem_ulonglong_g
#define shmem_longdouble_g           pshmem_longdouble_g
#define shmem_int8_g                 pshmem_int8_g
#define shmem_int16_g                pshmem_int16_g
#define shmem_int32_g                pshmem_int32_g
#define shmem_int64_g                pshmem_int64_g
#define shmem_uint8_g                pshmem_uint8_g
#define shmem_uint16_g               pshmem_uint16_g
#define shmem_uint32_g               pshmem_uint32_g
#define shmem_uint64_g               pshmem_uint64_g
#define shmem_size_g                 pshmem_size_g
#define shmem_ptrdiff_g              pshmem_ptrdiff_g

#define shmemx_int16_g               pshmemx_int16_g
#define shmemx_int32_g               pshmemx_int32_g
#define shmemx_int64_g               pshmemx_int64_g

/*
 * Block data get routines
 */
#define shmem_ctx_char_get           pshmem_ctx_char_get
#define shmem_ctx_short_get          pshmem_ctx_short_get
#define shmem_ctx_int_get            pshmem_ctx_int_get
#define shmem_ctx_long_get           pshmem_ctx_long_get
#define shmem_ctx_float_get          pshmem_ctx_float_get
#define shmem_ctx_double_get         pshmem_ctx_double_get
#define shmem_ctx_longlong_get       pshmem_ctx_longlong_get
#define shmem_ctx_schar_get          pshmem_ctx_schar_get
#define shmem_ctx_uchar_get          pshmem_ctx_uchar_get
#define shmem_ctx_ushort_get         pshmem_ctx_ushort_get
#define shmem_ctx_uint_get           pshmem_ctx_uint_get
#define shmem_ctx_ulong_get          pshmem_ctx_ulong_get
#define shmem_ctx_ulonglong_get      pshmem_ctx_ulonglong_get
#define shmem_ctx_longdouble_get     pshmem_ctx_longdouble_get
#define shmem_ctx_int8_get           pshmem_ctx_int8_get
#define shmem_ctx_int16_get          pshmem_ctx_int16_get
#define shmem_ctx_int32_get          pshmem_ctx_int32_get
#define shmem_ctx_int64_get          pshmem_ctx_int64_get
#define shmem_ctx_uint8_get          pshmem_ctx_uint8_get
#define shmem_ctx_uint16_get         pshmem_ctx_uint16_get
#define shmem_ctx_uint32_get         pshmem_ctx_uint32_get
#define shmem_ctx_uint64_get         pshmem_ctx_uint64_get
#define shmem_ctx_size_get           pshmem_ctx_size_get
#define shmem_ctx_ptrdiff_get        pshmem_ctx_ptrdiff_get

#define shmem_char_get               pshmem_char_get /* shmem-compat.h */
#define shmem_short_get              pshmem_short_get
#define shmem_int_get                pshmem_int_get
#define shmem_long_get               pshmem_long_get
#define shmem_float_get              pshmem_float_get
#define shmem_double_get             pshmem_double_get
#define shmem_longlong_get           pshmem_longlong_get
#define shmem_schar_get              pshmem_schar_get
#define shmem_uchar_get              pshmem_uchar_get
#define shmem_ushort_get             pshmem_ushort_get
#define shmem_uint_get               pshmem_uint_get
#define shmem_ulong_get              pshmem_ulong_get
#define shmem_ulonglong_get          pshmem_ulonglong_get
#define shmem_longdouble_get         pshmem_longdouble_get
#define shmem_int8_get               pshmem_int8_get
#define shmem_int16_get              pshmem_int16_get
#define shmem_int32_get              pshmem_int32_get
#define shmem_int64_get              pshmem_int64_get
#define shmem_uint8_get              pshmem_uint8_get
#define shmem_uint16_get             pshmem_uint16_get
#define shmem_uint32_get             pshmem_uint32_get
#define shmem_uint64_get             pshmem_uint64_get
#define shmem_size_get               pshmem_size_get
#define shmem_ptrdiff_get            pshmem_ptrdiff_get

#define shmem_ctx_get8               pshmem_ctx_get8
#define shmem_ctx_get16              pshmem_ctx_get16
#define shmem_ctx_get32              pshmem_ctx_get32
#define shmem_ctx_get64              pshmem_ctx_get64
#define shmem_ctx_get128             pshmem_ctx_get128
#define shmem_ctx_getmem             pshmem_ctx_getmem

#define shmem_get8                   pshmem_get8
#define shmem_get16                  pshmem_get16
#define shmem_get32                  pshmem_get32
#define shmem_get64                  pshmem_get64
#define shmem_get128                 pshmem_get128
#define shmem_getmem                 pshmem_getmem

/*
 * Strided get routines
 */
#define shmem_ctx_char_iget           pshmem_ctx_char_iget
#define shmem_ctx_short_iget          pshmem_ctx_short_iget
#define shmem_ctx_int_iget            pshmem_ctx_int_iget
#define shmem_ctx_long_iget           pshmem_ctx_long_iget
#define shmem_ctx_float_iget          pshmem_ctx_float_iget
#define shmem_ctx_double_iget         pshmem_ctx_double_iget
#define shmem_ctx_longlong_iget       pshmem_ctx_longlong_iget
#define shmem_ctx_schar_iget          pshmem_ctx_schar_iget
#define shmem_ctx_uchar_iget          pshmem_ctx_uchar_iget
#define shmem_ctx_ushort_iget         pshmem_ctx_ushort_iget
#define shmem_ctx_uint_iget           pshmem_ctx_uint_iget
#define shmem_ctx_ulong_iget          pshmem_ctx_ulong_iget
#define shmem_ctx_ulonglong_iget      pshmem_ctx_ulonglong_iget
#define shmem_ctx_longdouble_iget     pshmem_ctx_longdouble_iget
#define shmem_ctx_int8_iget           pshmem_ctx_int8_iget
#define shmem_ctx_int16_iget          pshmem_ctx_int16_iget
#define shmem_ctx_int32_iget          pshmem_ctx_int32_iget
#define shmem_ctx_int64_iget          pshmem_ctx_int64_iget
#define shmem_ctx_uint8_iget          pshmem_ctx_uint8_iget
#define shmem_ctx_uint16_iget         pshmem_ctx_uint16_iget
#define shmem_ctx_uint32_iget         pshmem_ctx_uint32_iget
#define shmem_ctx_uint64_iget         pshmem_ctx_uint64_iget
#define shmem_ctx_size_iget           pshmem_ctx_size_iget
#define shmem_ctx_ptrdiff_iget        pshmem_ctx_ptrdiff_iget

#define shmem_char_iget               pshmem_char_iget
#define shmem_short_iget              pshmem_short_iget
#define shmem_int_iget                pshmem_int_iget
#define shmem_long_iget               pshmem_long_iget
#define shmem_float_iget              pshmem_float_iget
#define shmem_double_iget             pshmem_double_iget
#define shmem_longlong_iget           pshmem_longlong_iget
#define shmem_schar_iget              pshmem_schar_iget
#define shmem_uchar_iget              pshmem_uchar_iget
#define shmem_ushort_iget             pshmem_ushort_iget
#define shmem_uint_iget               pshmem_uint_iget
#define shmem_ulong_iget              pshmem_ulong_iget
#define shmem_ulonglong_iget          pshmem_ulonglong_iget
#define shmem_longdouble_iget         pshmem_longdouble_iget
#define shmem_int8_iget               pshmem_int8_iget
#define shmem_int16_iget              pshmem_int16_iget
#define shmem_int32_iget              pshmem_int32_iget
#define shmem_int64_iget              pshmem_int64_iget
#define shmem_uint8_iget              pshmem_uint8_iget
#define shmem_uint16_iget             pshmem_uint16_iget
#define shmem_uint32_iget             pshmem_uint32_iget
#define shmem_uint64_iget             pshmem_uint64_iget
#define shmem_size_iget               pshmem_size_iget
#define shmem_ptrdiff_iget            pshmem_ptrdiff_iget

#define shmem_ctx_iget8              pshmem_ctx_iget8
#define shmem_ctx_iget16             pshmem_ctx_iget16
#define shmem_ctx_iget32             pshmem_ctx_iget32
#define shmem_ctx_iget64             pshmem_ctx_iget64
#define shmem_ctx_iget128            pshmem_ctx_iget128

#define shmem_iget8                  pshmem_iget8
#define shmem_iget16                 pshmem_iget16
#define shmem_iget32                 pshmem_iget32
#define shmem_iget64                 pshmem_iget64
#define shmem_iget128                pshmem_iget128

/*
 * Non-block data get routines
 */
#define shmem_ctx_char_get_nbi           pshmem_ctx_char_get_nbi
#define shmem_ctx_short_get_nbi          pshmem_ctx_short_get_nbi
#define shmem_ctx_int_get_nbi            pshmem_ctx_int_get_nbi
#define shmem_ctx_long_get_nbi           pshmem_ctx_long_get_nbi
#define shmem_ctx_float_get_nbi          pshmem_ctx_float_get_nbi
#define shmem_ctx_double_get_nbi         pshmem_ctx_double_get_nbi
#define shmem_ctx_longlong_get_nbi       pshmem_ctx_longlong_get_nbi
#define shmem_ctx_schar_get_nbi          pshmem_ctx_schar_get_nbi
#define shmem_ctx_uchar_get_nbi          pshmem_ctx_uchar_get_nbi
#define shmem_ctx_ushort_get_nbi         pshmem_ctx_ushort_get_nbi
#define shmem_ctx_uint_get_nbi           pshmem_ctx_uint_get_nbi
#define shmem_ctx_ulong_get_nbi          pshmem_ctx_ulong_get_nbi
#define shmem_ctx_ulonglong_get_nbi      pshmem_ctx_ulonglong_get_nbi
#define shmem_ctx_longdouble_get_nbi     pshmem_ctx_longdouble_get_nbi
#define shmem_ctx_int8_get_nbi           pshmem_ctx_int8_get_nbi
#define shmem_ctx_int16_get_nbi          pshmem_ctx_int16_get_nbi
#define shmem_ctx_int32_get_nbi          pshmem_ctx_int32_get_nbi
#define shmem_ctx_int64_get_nbi          pshmem_ctx_int64_get_nbi
#define shmem_ctx_uint8_get_nbi          pshmem_ctx_uint8_get_nbi
#define shmem_ctx_uint16_get_nbi         pshmem_ctx_uint16_get_nbi
#define shmem_ctx_uint32_get_nbi         pshmem_ctx_uint32_get_nbi
#define shmem_ctx_uint64_get_nbi         pshmem_ctx_uint64_get_nbi
#define shmem_ctx_size_get_nbi           pshmem_ctx_size_get_nbi
#define shmem_ctx_ptrdiff_get_nbi        pshmem_ctx_ptrdiff_get_nbi

#define shmem_char_get_nbi               pshmem_char_get_nbi
#define shmem_short_get_nbi              pshmem_short_get_nbi
#define shmem_int_get_nbi                pshmem_int_get_nbi
#define shmem_long_get_nbi               pshmem_long_get_nbi
#define shmem_float_get_nbi              pshmem_float_get_nbi
#define shmem_double_get_nbi             pshmem_double_get_nbi
#define shmem_longlong_get_nbi           pshmem_longlong_get_nbi
#define shmem_schar_get_nbi              pshmem_schar_get_nbi
#define shmem_uchar_get_nbi              pshmem_uchar_get_nbi
#define shmem_ushort_get_nbi             pshmem_ushort_get_nbi
#define shmem_uint_get_nbi               pshmem_uint_get_nbi
#define shmem_ulong_get_nbi              pshmem_ulong_get_nbi
#define shmem_ulonglong_get_nbi          pshmem_ulonglong_get_nbi
#define shmem_longdouble_get_nbi         pshmem_longdouble_get_nbi
#define shmem_int8_get_nbi               pshmem_int8_get_nbi
#define shmem_int16_get_nbi              pshmem_int16_get_nbi
#define shmem_int32_get_nbi              pshmem_int32_get_nbi
#define shmem_int64_get_nbi              pshmem_int64_get_nbi
#define shmem_uint8_get_nbi              pshmem_uint8_get_nbi
#define shmem_uint16_get_nbi             pshmem_uint16_get_nbi
#define shmem_uint32_get_nbi             pshmem_uint32_get_nbi
#define shmem_uint64_get_nbi             pshmem_uint64_get_nbi
#define shmem_size_get_nbi               pshmem_size_get_nbi
#define shmem_ptrdiff_get_nbi            pshmem_ptrdiff_get_nbi

#define shmem_ctx_get8_nbi           pshmem_ctx_get8_nbi
#define shmem_ctx_get16_nbi          pshmem_ctx_get16_nbi
#define shmem_ctx_get32_nbi          pshmem_ctx_get32_nbi
#define shmem_ctx_get64_nbi          pshmem_ctx_get64_nbi
#define shmem_ctx_get128_nbi         pshmem_ctx_get128_nbi
#define shmem_ctx_getmem_nbi         pshmem_ctx_getmem_nbi

#define shmem_get8_nbi               pshmem_get8_nbi
#define shmem_get16_nbi              pshmem_get16_nbi
#define shmem_get32_nbi              pshmem_get32_nbi
#define shmem_get64_nbi              pshmem_get64_nbi
#define shmem_get128_nbi             pshmem_get128_nbi
#define shmem_getmem_nbi             pshmem_getmem_nbi

/*
 * Atomic operations
 */
/* Atomic swap */
#define shmem_ctx_double_atomic_swap pshmem_ctx_double_atomic_swap
#define shmem_ctx_float_atomic_swap  pshmem_ctx_float_atomic_swap
#define shmem_ctx_int_atomic_swap    pshmem_ctx_int_atomic_swap
#define shmem_ctx_long_atomic_swap   pshmem_ctx_long_atomic_swap
#define shmem_ctx_longlong_atomic_swap pshmem_ctx_longlong_atomic_swap
#define shmem_ctx_uint_atomic_swap   pshmem_ctx_uint_atomic_swap
#define shmem_ctx_ulong_atomic_swap  pshmem_ctx_ulong_atomic_swap
#define shmem_ctx_ulonglong_atomic_swap pshmem_ctx_ulonglong_atomic_swap

#define shmem_double_atomic_swap     pshmem_double_atomic_swap
#define shmem_float_atomic_swap      pshmem_float_atomic_swap
#define shmem_int_atomic_swap        pshmem_int_atomic_swap
#define shmem_long_atomic_swap       pshmem_long_atomic_swap
#define shmem_longlong_atomic_swap   pshmem_longlong_atomic_swap
#define shmem_uint_atomic_swap       pshmem_uint_atomic_swap
#define shmem_ulong_atomic_swap      pshmem_ulong_atomic_swap
#define shmem_ulonglong_atomic_swap  pshmem_ulonglong_atomic_swap

#define shmem_double_swap            pshmem_double_swap
#define shmem_float_swap             pshmem_float_swap
#define shmem_int_swap               pshmem_int_swap
#define shmem_long_swap              pshmem_long_swap
#define shmem_longlong_swap          pshmem_longlong_swap

#define shmemx_int32_swap            pshmemx_int32_swap
#define shmemx_int64_swap            pshmemx_int64_swap

/* Atomic set */
#define shmem_ctx_double_atomic_set pshmem_ctx_double_atomic_set
#define shmem_ctx_float_atomic_set  pshmem_ctx_float_atomic_set
#define shmem_ctx_int_atomic_set    pshmem_ctx_int_atomic_set
#define shmem_ctx_long_atomic_set   pshmem_ctx_long_atomic_set
#define shmem_ctx_longlong_atomic_set pshmem_ctx_longlong_atomic_set
#define shmem_ctx_uint_atomic_set   pshmem_ctx_uint_atomic_set
#define shmem_ctx_ulong_atomic_set  pshmem_ctx_ulong_atomic_set
#define shmem_ctx_ulonglong_atomic_set pshmem_ctx_ulonglong_atomic_set

#define shmem_double_atomic_set     pshmem_double_atomic_set
#define shmem_float_atomic_set      pshmem_float_atomic_set
#define shmem_int_atomic_set        pshmem_int_atomic_set
#define shmem_long_atomic_set       pshmem_long_atomic_set
#define shmem_longlong_atomic_set   pshmem_longlong_atomic_set
#define shmem_uint_atomic_set       pshmem_uint_atomic_set
#define shmem_ulong_atomic_set      pshmem_ulong_atomic_set
#define shmem_ulonglong_atomic_set  pshmem_ulonglong_atomic_set

#define shmem_double_set            pshmem_double_set
#define shmem_float_set             pshmem_float_set
#define shmem_int_set               pshmem_int_set
#define shmem_long_set              pshmem_long_set
#define shmem_longlong_set          pshmem_longlong_set

#define shmemx_int32_set            pshmemx_int32_set
#define shmemx_int64_set            pshmemx_int64_set

/* Atomic conditional swap */
#define shmem_ctx_int_atomic_compare_swap   pshmem_ctx_int_atomic_compare_swap
#define shmem_ctx_long_atomic_compare_swap  pshmem_ctx_long_atomic_compare_swap
#define shmem_ctx_longlong_atomic_compare_swap pshmem_ctx_longlong_atomic_compare_swap
#define shmem_ctx_uint_atomic_compare_swap  pshmem_ctx_uint_atomic_compare_swap
#define shmem_ctx_ulong_atomic_compare_swap pshmem_ctx_ulong_atomic_compare_swap
#define shmem_ctx_ulonglong_atomic_compare_swap pshmem_ctx_ulonglong_atomic_compare_swap

#define shmem_int_atomic_compare_swap       pshmem_int_atomic_compare_swap
#define shmem_long_atomic_compare_swap      pshmem_long_atomic_compare_swap
#define shmem_longlong_atomic_compare_swap  pshmem_longlong_atomic_compare_swap
#define shmem_uint_atomic_compare_swap      pshmem_uint_atomic_compare_swap
#define shmem_ulong_atomic_compare_swap     pshmem_ulong_atomic_compare_swap
#define shmem_ulonglong_atomic_compare_swap pshmem_ulonglong_atomic_compare_swap

#define shmem_int_cswap              pshmem_int_cswap
#define shmem_long_cswap             pshmem_long_cswap
#define shmem_longlong_cswap         pshmem_longlong_cswap

#define shmemx_int32_cswap           pshmemx_int32_cswap
#define shmemx_int64_cswap           pshmemx_int64_cswap

/* Atomic Fetch&Add */
#define shmem_ctx_int_atomic_fetch_add       pshmem_ctx_int_atomic_fetch_add
#define shmem_ctx_long_atomic_fetch_add      pshmem_ctx_long_atomic_fetch_add
#define shmem_ctx_longlong_atomic_fetch_add  pshmem_ctx_longlong_atomic_fetch_add
#define shmem_ctx_uint_atomic_fetch_add      pshmem_ctx_uint_atomic_fetch_add
#define shmem_ctx_ulong_atomic_fetch_add     pshmem_ctx_ulong_atomic_fetch_add
#define shmem_ctx_ulonglong_atomic_fetch_add pshmem_ctx_ulonglong_atomic_fetch_add

#define shmem_int_atomic_fetch_add           pshmem_int_atomic_fetch_add
#define shmem_long_atomic_fetch_add          pshmem_long_atomic_fetch_add
#define shmem_longlong_atomic_fetch_add      pshmem_longlong_atomic_fetch_add
#define shmem_uint_atomic_fetch_add          pshmem_uint_atomic_fetch_add
#define shmem_ulong_atomic_fetch_add         pshmem_ulong_atomic_fetch_add
#define shmem_ulonglong_atomic_fetch_add     pshmem_ulonglong_atomic_fetch_add

#define shmem_int_fadd                       pshmem_int_fadd
#define shmem_long_fadd                      pshmem_long_fadd
#define shmem_longlong_fadd                  pshmem_longlong_fadd

#define shmemx_int32_fadd                    pshmemx_int32_fadd
#define shmemx_int64_fadd                    pshmemx_int64_fadd

/* Atomic Fetch&And */
#define shmem_int_atomic_fetch_and        pshmem_int_atomic_fetch_and
#define shmem_long_atomic_fetch_and       pshmem_long_atomic_fetch_and
#define shmem_longlong_atomic_fetch_and   pshmem_longlong_atomic_fetch_and
#define shmem_uint_atomic_fetch_and       pshmem_uint_atomic_fetch_and
#define shmem_ulong_atomic_fetch_and      pshmem_ulong_atomic_fetch_and
#define shmem_ulonglong_atomic_fetch_and  pshmem_ulonglong_atomic_fetch_and
#define shmem_int32_atomic_fetch_and      pshmem_int32_atomic_fetch_and
#define shmem_int64_atomic_fetch_and      pshmem_int64_atomic_fetch_and
#define shmem_uint32_atomic_fetch_and     pshmem_uint32_atomic_fetch_and
#define shmem_uint64_atomic_fetch_and     pshmem_uint64_atomic_fetch_and

#define shmem_ctx_int_atomic_fetch_and    pshmem_ctx_int_atomic_fetch_and
#define shmem_ctx_long_atomic_fetch_and   pshmem_ctx_long_atomic_fetch_and
#define shmem_ctx_longlong_atomic_fetch_and pshmem_ctx_longlong_atomic_fetch_and
#define shmem_ctx_uint_atomic_fetch_and   pshmem_ctx_uint_atomic_fetch_and
#define shmem_ctx_ulong_atomic_fetch_and  pshmem_ctx_ulong_atomic_fetch_and
#define shmem_ctx_ulonglong_atomic_fetch_and pshmem_ctx_ulonglong_atomic_fetch_and
#define shmem_ctx_int32_atomic_fetch_and  pshmem_ctx_int32_atomic_fetch_and
#define shmem_ctx_int64_atomic_fetch_and  pshmem_ctx_int64_atomic_fetch_and
#define shmem_ctx_uint32_atomic_fetch_and pshmem_ctx_uint32_atomic_fetch_and
#define shmem_ctx_uint64_atomic_fetch_and pshmem_ctx_uint64_atomic_fetch_and

#define shmemx_int32_atomic_fetch_and     pshmemx_int32_atomic_fetch_and
#define shmemx_int64_atomic_fetch_and     pshmemx_int64_atomic_fetch_and
#define shmemx_uint32_atomic_fetch_and    pshmemx_uint32_atomic_fetch_and
#define shmemx_uint64_atomic_fetch_and    pshmemx_uint64_atomic_fetch_and

/* Atomic Fetch&Or */
#define shmem_int_atomic_fetch_or         pshmem_int_atomic_fetch_or
#define shmem_long_atomic_fetch_or        pshmem_long_atomic_fetch_or
#define shmem_longlong_atomic_fetch_or    pshmem_longlong_atomic_fetch_or
#define shmem_uint_atomic_fetch_or        pshmem_uint_atomic_fetch_or
#define shmem_ulong_atomic_fetch_or       pshmem_ulong_atomic_fetch_or
#define shmem_ulonglong_atomic_fetch_or   pshmem_ulonglong_atomic_fetch_or
#define shmem_int32_atomic_fetch_or       pshmem_int32_atomic_fetch_or
#define shmem_int64_atomic_fetch_or       pshmem_int64_atomic_fetch_or
#define shmem_uint32_atomic_fetch_or      pshmem_uint32_atomic_fetch_or
#define shmem_uint64_atomic_fetch_or      pshmem_uint64_atomic_fetch_or

#define shmem_ctx_int_atomic_fetch_or     pshmem_ctx_int_atomic_fetch_or
#define shmem_ctx_long_atomic_fetch_or    pshmem_ctx_long_atomic_fetch_or
#define shmem_ctx_longlong_atomic_fetch_or pshmem_ctx_longlong_atomic_fetch_or
#define shmem_ctx_uint_atomic_fetch_or    pshmem_ctx_uint_atomic_fetch_or
#define shmem_ctx_ulong_atomic_fetch_or   pshmem_ctx_ulong_atomic_fetch_or
#define shmem_ctx_ulonglong_atomic_fetch_or pshmem_ctx_ulonglong_atomic_fetch_or
#define shmem_ctx_int32_atomic_fetch_or   pshmem_ctx_int32_atomic_fetch_or
#define shmem_ctx_int64_atomic_fetch_or   pshmem_ctx_int64_atomic_fetch_or
#define shmem_ctx_uint32_atomic_fetch_or  pshmem_ctx_uint32_atomic_fetch_or
#define shmem_ctx_uint64_atomic_fetch_or  pshmem_ctx_uint64_atomic_fetch_or

#define shmemx_int32_atomic_fetch_or      pshmemx_int32_atomic_fetch_or
#define shmemx_int64_atomic_fetch_or      pshmemx_int64_atomic_fetch_or
#define shmemx_uint32_atomic_fetch_or     pshmemx_uint32_atomic_fetch_or
#define shmemx_uint64_atomic_fetch_or     pshmemx_uint64_atomic_fetch_or

/* Atomic Fetch&Xor */
#define shmem_int_atomic_fetch_xor        pshmem_int_atomic_fetch_xor
#define shmem_long_atomic_fetch_xor       pshmem_long_atomic_fetch_xor
#define shmem_longlong_atomic_fetch_xor   pshmem_longlong_atomic_fetch_xor
#define shmem_uint_atomic_fetch_xor       pshmem_uint_atomic_fetch_xor
#define shmem_ulong_atomic_fetch_xor      pshmem_ulong_atomic_fetch_xor
#define shmem_ulonglong_atomic_fetch_xor  pshmem_ulonglong_atomic_fetch_xor
#define shmem_int32_atomic_fetch_xor      pshmem_int32_atomic_fetch_xor
#define shmem_int64_atomic_fetch_xor      pshmem_int64_atomic_fetch_xor
#define shmem_uint32_atomic_fetch_xor     pshmem_uint32_atomic_fetch_xor
#define shmem_uint64_atomic_fetch_xor     pshmem_uint64_atomic_fetch_xor

#define shmem_ctx_int_atomic_fetch_xor    pshmem_ctx_int_atomic_fetch_xor
#define shmem_ctx_long_atomic_fetch_xor   pshmem_ctx_long_atomic_fetch_xor
#define shmem_ctx_longlong_atomic_fetch_xor pshmem_ctx_longlong_atomic_fetch_xor
#define shmem_ctx_uint_atomic_fetch_xor   pshmem_ctx_uint_atomic_fetch_xor
#define shmem_ctx_ulong_atomic_fetch_xor  pshmem_ctx_ulong_atomic_fetch_xor
#define shmem_ctx_ulonglong_atomic_fetch_xor pshmem_ctx_ulonglong_atomic_fetch_xor
#define shmem_ctx_int32_atomic_fetch_xor  pshmem_ctx_int32_atomic_fetch_xor
#define shmem_ctx_int64_atomic_fetch_xor  pshmem_ctx_int64_atomic_fetch_xor
#define shmem_ctx_uint32_atomic_fetch_xor pshmem_ctx_uint32_atomic_fetch_xor
#define shmem_ctx_uint64_atomic_fetch_xor pshmem_ctx_uint64_atomic_fetch_xor

#define shmemx_int32_atomic_fetch_xor     pshmemx_int32_atomic_fetch_xor
#define shmemx_int64_atomic_fetch_xor     pshmemx_int64_atomic_fetch_xor
#define shmemx_uint32_atomic_fetch_xor    pshmemx_uint32_atomic_fetch_xor
#define shmemx_uint64_atomic_fetch_xor    pshmemx_uint64_atomic_fetch_xor

/* Atomic Fetch */
#define shmem_ctx_double_atomic_fetch pshmem_ctx_double_atomic_fetch
#define shmem_ctx_float_atomic_fetch  pshmem_ctx_float_atomic_fetch
#define shmem_ctx_int_atomic_fetch    pshmem_ctx_int_atomic_fetch
#define shmem_ctx_long_atomic_fetch   pshmem_ctx_long_atomic_fetch
#define shmem_ctx_longlong_atomic_fetch pshmem_ctx_longlong_atomic_fetch
#define shmem_ctx_uint_atomic_fetch   pshmem_ctx_uint_atomic_fetch
#define shmem_ctx_ulong_atomic_fetch  pshmem_ctx_ulong_atomic_fetch
#define shmem_ctx_ulonglong_atomic_fetch pshmem_ctx_ulonglong_atomic_fetch

#define shmem_double_atomic_fetch     pshmem_double_atomic_fetch
#define shmem_float_atomic_fetch      pshmem_float_atomic_fetch
#define shmem_int_atomic_fetch        pshmem_int_atomic_fetch
#define shmem_long_atomic_fetch       pshmem_long_atomic_fetch
#define shmem_longlong_atomic_fetch   pshmem_longlong_atomic_fetch
#define shmem_uint_atomic_fetch       pshmem_uint_atomic_fetch
#define shmem_ulong_atomic_fetch      pshmem_ulong_atomic_fetch
#define shmem_ulonglong_atomic_fetch  pshmem_ulonglong_atomic_fetch

#define shmem_double_fetch            pshmem_double_fetch
#define shmem_float_fetch             pshmem_float_fetch
#define shmem_int_fetch               pshmem_int_fetch
#define shmem_long_fetch              pshmem_long_fetch
#define shmem_longlong_fetch          pshmem_longlong_fetch

#define shmemx_int32_fetch            pshmemx_int32_fetch
#define shmemx_int64_fetch            pshmemx_int64_fetch

/* Atomic Fetch&Inc */
#define shmem_ctx_int_atomic_fetch_inc    pshmem_ctx_int_atomic_fetch_inc
#define shmem_ctx_long_atomic_fetch_inc   pshmem_ctx_long_atomic_fetch_inc
#define shmem_ctx_longlong_atomic_fetch_inc pshmem_ctx_longlong_atomic_fetch_inc
#define shmem_ctx_uint_atomic_fetch_inc    pshmem_ctx_uint_atomic_fetch_inc
#define shmem_ctx_ulong_atomic_fetch_inc   pshmem_ctx_ulong_atomic_fetch_inc
#define shmem_ctx_ulonglong_atomic_fetch_inc pshmem_ctx_ulonglong_atomic_fetch_inc

#define shmem_uint_atomic_fetch_inc        pshmem_uint_atomic_fetch_inc
#define shmem_ulong_atomic_fetch_inc       pshmem_ulong_atomic_fetch_inc
#define shmem_ulonglong_atomic_fetch_inc   pshmem_ulonglong_atomic_fetch_inc
#define shmem_int_atomic_fetch_inc        pshmem_int_atomic_fetch_inc
#define shmem_long_atomic_fetch_inc       pshmem_long_atomic_fetch_inc
#define shmem_longlong_atomic_fetch_inc   pshmem_longlong_atomic_fetch_inc

#define shmem_int_finc               pshmem_int_finc
#define shmem_long_finc              pshmem_long_finc
#define shmem_longlong_finc          pshmem_longlong_finc

#define shmemx_int32_finc            pshmemx_int32_finc
#define shmemx_int64_finc            pshmemx_int64_finc

/* Atomic Add */
#define shmem_ctx_int_atomic_add     pshmem_ctx_int_atomic_add
#define shmem_ctx_long_atomic_add    pshmem_ctx_long_atomic_add
#define shmem_ctx_longlong_atomic_add pshmem_ctx_longlong_atomic_add
#define shmem_ctx_uint_atomic_add    pshmem_ctx_uint_atomic_add
#define shmem_ctx_ulong_atomic_add   pshmem_ctx_ulong_atomic_add
#define shmem_ctx_ulonglong_atomic_add pshmem_ctx_ulonglong_atomic_add

#define shmem_int_atomic_add         pshmem_int_atomic_add
#define shmem_long_atomic_add        pshmem_long_atomic_add
#define shmem_longlong_atomic_add    pshmem_longlong_atomic_add
#define shmem_uint_atomic_add        pshmem_uint_atomic_add
#define shmem_ulong_atomic_add       pshmem_ulong_atomic_add
#define shmem_ulonglong_atomic_add   pshmem_ulonglong_atomic_add

#define shmem_int_add                pshmem_int_add
#define shmem_long_add               pshmem_long_add
#define shmem_longlong_add           pshmem_longlong_add

#define shmemx_int32_add             pshmemx_int32_add
#define shmemx_int64_add             pshmemx_int64_add

/* Atomic And */
#define shmem_int_atomic_and         pshmem_int_atomic_and
#define shmem_long_atomic_and        pshmem_long_atomic_and
#define shmem_longlong_atomic_and    pshmem_longlong_atomic_and
#define shmem_uint_atomic_and        pshmem_uint_atomic_and
#define shmem_ulong_atomic_and       pshmem_ulong_atomic_and
#define shmem_ulonglong_atomic_and   pshmem_ulonglong_atomic_and
#define shmem_int32_atomic_and       pshmem_int32_atomic_and
#define shmem_int64_atomic_and       pshmem_int64_atomic_and
#define shmem_uint32_atomic_and      pshmem_uint32_atomic_and
#define shmem_uint64_atomic_and      pshmem_uint64_atomic_and

#define shmem_ctx_int_atomic_and     pshmem_ctx_int_atomic_and
#define shmem_ctx_long_atomic_and    pshmem_ctx_long_atomic_and
#define shmem_ctx_longlong_atomic_and pshmem_ctx_longlong_atomic_and
#define shmem_ctx_uint_atomic_and    pshmem_ctx_uint_atomic_and
#define shmem_ctx_ulong_atomic_and   pshmem_ctx_ulong_atomic_and
#define shmem_ctx_ulonglong_atomic_and pshmem_ctx_ulonglong_atomic_and
#define shmem_ctx_int32_atomic_and   pshmem_ctx_int32_atomic_and
#define shmem_ctx_int64_atomic_and   pshmem_ctx_int64_atomic_and
#define shmem_ctx_uint32_atomic_and  pshmem_ctx_uint32_atomic_and
#define shmem_ctx_uint64_atomic_and  pshmem_ctx_uint64_atomic_and

#define shmemx_int32_atomic_and      pshmemx_int32_atomic_and
#define shmemx_int64_atomic_and      pshmemx_int64_atomic_and

#define shmemx_uint32_atomic_and     pshmemx_uint32_atomic_and
#define shmemx_uint64_atomic_and     pshmemx_uint64_atomic_and

/* Atomic Or */
#define shmem_int_atomic_or          pshmem_int_atomic_or
#define shmem_long_atomic_or         pshmem_long_atomic_or
#define shmem_longlong_atomic_or     pshmem_longlong_atomic_or
#define shmem_uint_atomic_or         pshmem_uint_atomic_or
#define shmem_ulong_atomic_or        pshmem_ulong_atomic_or
#define shmem_ulonglong_atomic_or    pshmem_ulonglong_atomic_or
#define shmem_int32_atomic_or        pshmem_int32_atomic_or
#define shmem_int64_atomic_or        pshmem_int64_atomic_or
#define shmem_uint32_atomic_or       pshmem_uint32_atomic_or
#define shmem_uint64_atomic_or       pshmem_uint64_atomic_or

#define shmem_ctx_int_atomic_or      pshmem_ctx_int_atomic_or
#define shmem_ctx_long_atomic_or     pshmem_ctx_long_atomic_or
#define shmem_ctx_longlong_atomic_or pshmem_ctx_longlong_atomic_or
#define shmem_ctx_uint_atomic_or     pshmem_ctx_uint_atomic_or
#define shmem_ctx_ulong_atomic_or    pshmem_ctx_ulong_atomic_or
#define shmem_ctx_ulonglong_atomic_or pshmem_ctx_ulonglong_atomic_or
#define shmem_ctx_int32_atomic_or    pshmem_ctx_int32_atomic_or
#define shmem_ctx_int64_atomic_or    pshmem_ctx_int64_atomic_or
#define shmem_ctx_uint32_atomic_or   pshmem_ctx_uint32_atomic_or
#define shmem_ctx_uint64_atomic_or   pshmem_ctx_uint64_atomic_or

#define shmemx_int32_atomic_or       pshmemx_int32_atomic_or
#define shmemx_int64_atomic_or       pshmemx_int64_atomic_or

#define shmemx_uint32_atomic_or      pshmemx_uint32_atomic_or
#define shmemx_uint64_atomic_or      pshmemx_uint64_atomic_or

/* Atomic Xor */
#define shmem_int_atomic_xor         pshmem_int_atomic_xor
#define shmem_long_atomic_xor        pshmem_long_atomic_xor
#define shmem_longlong_atomic_xor    pshmem_longlong_atomic_xor
#define shmem_uint_atomic_xor        pshmem_uint_atomic_xor
#define shmem_ulong_atomic_xor       pshmem_ulong_atomic_xor
#define shmem_ulonglong_atomic_xor   pshmem_ulonglong_atomic_xor
#define shmem_int32_atomic_xor       pshmem_int32_atomic_xor
#define shmem_int64_atomic_xor       pshmem_int64_atomic_xor
#define shmem_uint32_atomic_xor      pshmem_uint32_atomic_xor
#define shmem_uint64_atomic_xor      pshmem_uint64_atomic_xor

#define shmem_ctx_int_atomic_xor     pshmem_ctx_int_atomic_xor
#define shmem_ctx_long_atomic_xor    pshmem_ctx_long_atomic_xor
#define shmem_ctx_longlong_atomic_xor pshmem_ctx_longlong_atomic_xor
#define shmem_ctx_uint_atomic_xor    pshmem_ctx_uint_atomic_xor
#define shmem_ctx_ulong_atomic_xor   pshmem_ctx_ulong_atomic_xor
#define shmem_ctx_ulonglong_atomic_xor pshmem_ctx_ulonglong_atomic_xor
#define shmem_ctx_int32_atomic_xor   pshmem_ctx_int32_atomic_xor
#define shmem_ctx_int64_atomic_xor   pshmem_ctx_int64_atomic_xor
#define shmem_ctx_uint32_atomic_xor  pshmem_ctx_uint32_atomic_xor
#define shmem_ctx_uint64_atomic_xor  pshmem_ctx_uint64_atomic_xor

#define shmemx_int32_atomic_xor      pshmemx_int32_atomic_xor
#define shmemx_int64_atomic_xor      pshmemx_int64_atomic_xor

#define shmemx_uint32_atomic_xor     pshmemx_uint32_atomic_xor
#define shmemx_uint64_atomic_xor     pshmemx_uint64_atomic_xor

/* Atomic Inc */
#define shmem_ctx_int_atomic_inc     pshmem_ctx_int_atomic_inc
#define shmem_ctx_long_atomic_inc    pshmem_ctx_long_atomic_inc
#define shmem_ctx_longlong_atomic_inc pshmem_ctx_longlong_atomic_inc
#define shmem_ctx_uint_atomic_inc    pshmem_ctx_uint_atomic_inc
#define shmem_ctx_ulong_atomic_inc   pshmem_ctx_ulong_atomic_inc
#define shmem_ctx_ulonglong_atomic_inc pshmem_ctx_ulonglong_atomic_inc

#define shmem_int_atomic_inc         pshmem_int_atomic_inc
#define shmem_long_atomic_inc        pshmem_long_atomic_inc
#define shmem_longlong_atomic_inc    pshmem_longlong_atomic_inc
#define shmem_uint_atomic_inc        pshmem_uint_atomic_inc
#define shmem_ulong_atomic_inc       pshmem_ulong_atomic_inc
#define shmem_ulonglong_atomic_inc   pshmem_ulonglong_atomic_inc

#define shmem_int_inc                pshmem_int_inc
#define shmem_long_inc               pshmem_long_inc
#define shmem_longlong_inc           pshmem_longlong_inc

#define shmemx_int32_inc             pshmemx_int32_inc
#define shmemx_int64_inc             pshmemx_int64_inc

/*
 * Lock functions
 */
#define shmem_set_lock               pshmem_set_lock
#define shmem_clear_lock             pshmem_clear_lock
#define shmem_test_lock              pshmem_test_lock

/*
 * P2P sync routines
 */
#define shmem_short_wait             pshmem_short_wait
#define shmem_int_wait               pshmem_int_wait
#define shmem_long_wait              pshmem_long_wait
#define shmem_longlong_wait          pshmem_longlong_wait
#define shmem_wait                   pshmem_wait
#define shmemx_int32_wait            pshmemx_int32_wait
#define shmemx_int64_wait            pshmemx_int64_wait

#define shmem_short_wait_until       pshmem_short_wait_until
#define shmem_int_wait_until         pshmem_int_wait_until
#define shmem_long_wait_until        pshmem_long_wait_until
#define shmem_longlong_wait_until    pshmem_longlong_wait_until
#define shmem_ushort_wait_until      pshmem_ushort_wait_until
#define shmem_uint_wait_until        pshmem_uint_wait_until
#define shmem_ulong_wait_until       pshmem_ulong_wait_until
#define shmem_ulonglong_wait_until   pshmem_ulonglong_wait_until
#define shmem_int32_wait_until       pshmem_int32_wait_until
#define shmem_int64_wait_until       pshmem_int64_wait_until
#define shmem_uint32_wait_until      pshmem_uint32_wait_until
#define shmem_uint64_wait_until      pshmem_uint64_wait_until
#define shmem_size_wait_until        pshmem_size_wait_until
#define shmem_ptrdiff_wait_until     pshmem_ptrdiff_wait_until

#define shmemx_int32_wait_until      pshmemx_int32_wait_until
#define shmemx_int64_wait_until      pshmemx_int64_wait_until

#define shmem_short_test             pshmem_short_test
#define shmem_int_test               pshmem_int_test
#define shmem_long_test              pshmem_long_test
#define shmem_longlong_test          pshmem_longlong_test
#define shmem_ushort_test            pshmem_ushort_test
#define shmem_uint_test              pshmem_uint_test
#define shmem_ulong_test             pshmem_ulong_test
#define shmem_ulonglong_test         pshmem_ulonglong_test
#define shmem_int32_test             pshmem_int32_test
#define shmem_int64_test             pshmem_int64_test
#define shmem_uint32_test            pshmem_uint32_test
#define shmem_uint64_test            pshmem_uint64_test
#define shmem_size_test              pshmem_size_test
#define shmem_ptrdiff_test           pshmem_ptrdiff_test

/*
 * Barrier sync routines
 */
#define shmem_barrier                pshmem_barrier
#define shmem_barrier_all            pshmem_barrier_all
#define shmem_sync                   pshmem_sync
#define shmem_sync_all               pshmem_sync_all
#define shmem_fence                  pshmem_fence
#define shmem_ctx_fence              pshmem_ctx_fence
#define shmem_quiet                  pshmem_quiet
#define shmem_ctx_quiet              pshmem_ctx_quiet

/*
 * Collective routines
 */
#define shmem_broadcast32            pshmem_broadcast32
#define shmem_broadcast64            pshmem_broadcast64
#define shmem_collect32              pshmem_collect32
#define shmem_collect64              pshmem_collect64
#define shmem_fcollect32             pshmem_fcollect32
#define shmem_fcollect64             pshmem_fcollect64

/*
 * Reduction routines
 */
#define shmem_short_and_to_all       pshmem_short_and_to_all
#define shmem_int_and_to_all         pshmem_int_and_to_all
#define shmem_long_and_to_all        pshmem_long_and_to_all
#define shmem_longlong_and_to_all    pshmem_longlong_and_to_all
#define shmemx_int16_and_to_all      pshmemx_int16_and_to_all
#define shmemx_int32_and_to_all      pshmemx_int32_and_to_all
#define shmemx_int64_and_to_all      pshmemx_int64_and_to_all

#define shmem_short_or_to_all        pshmem_short_or_to_all
#define shmem_int_or_to_all          pshmem_int_or_to_all
#define shmem_long_or_to_all         pshmem_long_or_to_all
#define shmem_longlong_or_to_all     pshmem_longlong_or_to_all
#define shmemx_int16_or_to_all       pshmemx_int16_or_to_all
#define shmemx_int32_or_to_all       pshmemx_int32_or_to_all
#define shmemx_int64_or_to_all       pshmemx_int64_or_to_all

#define shmem_short_xor_to_all       pshmem_short_xor_to_all
#define shmem_int_xor_to_all         pshmem_int_xor_to_all
#define shmem_long_xor_to_all        pshmem_long_xor_to_all
#define shmem_longlong_xor_to_all    pshmem_longlong_xor_to_all
#define shmemx_int16_xor_to_all      pshmemx_int16_xor_to_all
#define shmemx_int32_xor_to_all      pshmemx_int32_xor_to_all
#define shmemx_int64_xor_to_all      pshmemx_int64_xor_to_all

#define shmem_short_max_to_all       pshmem_short_max_to_all
#define shmem_int_max_to_all         pshmem_int_max_to_all
#define shmem_long_max_to_all        pshmem_long_max_to_all
#define shmem_longlong_max_to_all    pshmem_longlong_max_to_all
#define shmem_float_max_to_all       pshmem_float_max_to_all
#define shmem_double_max_to_all      pshmem_double_max_to_all
#define shmem_longdouble_max_to_all  pshmem_longdouble_max_to_all
#define shmemx_int16_max_to_all      pshmemx_int16_max_to_all
#define shmemx_int32_max_to_all      pshmemx_int32_max_to_all
#define shmemx_int64_max_to_all      pshmemx_int64_max_to_all

#define shmem_short_min_to_all       pshmem_short_min_to_all
#define shmem_int_min_to_all         pshmem_int_min_to_all
#define shmem_long_min_to_all        pshmem_long_min_to_all
#define shmem_longlong_min_to_all    pshmem_longlong_min_to_all
#define shmem_float_min_to_all       pshmem_float_min_to_all
#define shmem_double_min_to_all      pshmem_double_min_to_all
#define shmem_longdouble_min_to_all  pshmem_longdouble_min_to_all
#define shmemx_int16_min_to_all      pshmemx_int16_min_to_all
#define shmemx_int32_min_to_all      pshmemx_int32_min_to_all
#define shmemx_int64_min_to_all      pshmemx_int64_min_to_all

#define shmem_short_sum_to_all       pshmem_short_sum_to_all
#define shmem_int_sum_to_all         pshmem_int_sum_to_all
#define shmem_long_sum_to_all        pshmem_long_sum_to_all
#define shmem_longlong_sum_to_all    pshmem_longlong_sum_to_all
#define shmem_float_sum_to_all       pshmem_float_sum_to_all
#define shmem_double_sum_to_all      pshmem_double_sum_to_all
#define shmem_longdouble_sum_to_all  pshmem_longdouble_sum_to_all
#define shmem_complexf_sum_to_all    pshmem_complexf_sum_to_all
#define shmem_complexd_sum_to_all    pshmem_complexd_sum_to_all
#define shmemx_int16_sum_to_all      pshmemx_int16_sum_to_all
#define shmemx_int32_sum_to_all      pshmemx_int32_sum_to_all
#define shmemx_int64_sum_to_all      pshmemx_int64_sum_to_all

#define shmem_short_prod_to_all      pshmem_short_prod_to_all
#define shmem_int_prod_to_all        pshmem_int_prod_to_all
#define shmem_long_prod_to_all       pshmem_long_prod_to_all
#define shmem_longlong_prod_to_all   pshmem_longlong_prod_to_all
#define shmem_float_prod_to_all      pshmem_float_prod_to_all
#define shmem_double_prod_to_all     pshmem_double_prod_to_all
#define shmem_longdouble_prod_to_all pshmem_longdouble_prod_to_all
#define shmem_complexf_prod_to_all   pshmem_complexf_prod_to_all
#define shmem_complexd_prod_to_all   pshmem_complexd_prod_to_all
#define shmemx_int16_prod_to_all     pshmemx_int16_prod_to_all
#define shmemx_int32_prod_to_all     pshmemx_int32_prod_to_all
#define shmemx_int64_prod_to_all     pshmemx_int64_prod_to_all

/*
 * Alltoall routines
 */
#define shmem_alltoall32             pshmem_alltoall32
#define shmem_alltoall64             pshmem_alltoall64
#define shmem_alltoalls32            pshmem_alltoalls32
#define shmem_alltoalls64            pshmem_alltoalls64

/*
 * Platform specific cache management routines
 */
#define shmem_udcflush              pshmem_udcflush
#define shmem_udcflush_line         pshmem_udcflush_line
#define shmem_set_cache_inv         pshmem_set_cache_inv
#define shmem_set_cache_line_inv    pshmem_set_cache_line_inv
#define shmem_clear_cache_inv       pshmem_clear_cache_inv
#define shmem_clear_cache_line_inv  pshmem_clear_cache_line_inv

#endif /* OSHMEM_C_PROFILE_DEFINES_H */
