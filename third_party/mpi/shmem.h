/* oshmem/include/shmem.h.  Generated from shmem.h.in by configure.  */
/*
 * Copyright (c) 2014-2018 Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2014      Intel, Inc. All rights reserved
 * Copyright (c) 2016      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2017      Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OSHMEM_SHMEM_H
#define OSHMEM_SHMEM_H


#include <stddef.h>     /* include for ptrdiff_t */
#include <stdint.h>     /* include for fixed width types */
#if defined(c_plusplus) || defined(__cplusplus)
#    include <complex>
#    define OSHMEM_COMPLEX_TYPE(type)    std::complex<type>
#else
#    include <complex.h>
#    define OSHMEM_COMPLEX_TYPE(type)    type complex
#endif

/*
 * SHMEM version
 */
#define OSHMEM_MAJOR_VERSION 4
#define OSHMEM_MINOR_VERSION 1
#define OSHMEM_RELEASE_VERSION 4


#ifndef OSHMEM_DECLSPEC
#  if defined(OPAL_C_HAVE_VISIBILITY) && (OPAL_C_HAVE_VISIBILITY == 1)
#     define OSHMEM_DECLSPEC __attribute__((visibility("default")))
#  else
#     define OSHMEM_DECLSPEC
#  endif
#endif

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define OSHMEM_HAVE_C11 1
#else
#define OSHMEM_HAVE_C11 0
#endif

#include <shmem-compat.h>

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

#if OSHMEM_HAVE_C11
#define __OSHMEM_VAR_ARG1_EXPAND(_arg1, ...) _arg1
#define __OSHMEM_VAR_ARG1(...) __OSHMEM_VAR_ARG1_EXPAND(__VA_ARGS__, _extra)
#define __OSHMEM_VAR_ARG2(_arg1, ...) __OSHMEM_VAR_ARG1_EXPAND(__VA_ARGS__, _extra)
static inline void __oshmem_datatype_ignore(void) {}
#endif

/*
 * SHMEM_Init_thread constants
 */
enum {
    SHMEM_THREAD_SINGLE,
    SHMEM_THREAD_FUNNELED,
    SHMEM_THREAD_SERIALIZED,
    SHMEM_THREAD_MULTIPLE
};

/*
 * OpenSHMEM API (www.openshmem.org)
 */

/*
 * Environment variables
 */

/* Following environment variables are Mellanox extension */

/* size of symmetric heap in bytes.
 * Can be qualified with the letter 'K', 'M', 'G' or 'T'
 */
#define SHMEM_HEAP_SIZE "SHMEM_SYMMETRIC_HEAP_SIZE"

/*
 * Type of allocator used by symmetric heap
 */
#define SHMEM_HEAP_TYPE "SHMEM_SYMMETRIC_HEAP_ALLOCATOR"

/*
 * Constants and definitions
 */
#define SHMEM_MAJOR_VERSION             1
#define SHMEM_MINOR_VERSION             4 
#define SHMEM_VENDOR_STRING             "http://www.open-mpi.org/"
#define SHMEM_MAX_NAME_LEN              256

#define SHMEM_CTX_PRIVATE               (1<<0)
#define SHMEM_CTX_SERIALIZED            (1<<1)
#define SHMEM_CTX_NOSTORE               (1<<2)

/*
 * Deprecated (but still valid) names
 */
#define _SHMEM_MAJOR_VERSION            SHMEM_MAJOR_VERSION
#define _SHMEM_MINOR_VERSION            SHMEM_MINOR_VERSION
#define _SHMEM_MAX_NAME_LEN             SHMEM_MAX_NAME_LEN

#ifndef OSHMEM_SPEC_VERSION
#define OSHMEM_SPEC_VERSION (SHMEM_MAJOR_VERSION * 10000 + SHMEM_MINOR_VERSION * 100)
#endif

enum shmem_wait_ops {
    SHMEM_CMP_EQ,
    SHMEM_CMP_NE,
    SHMEM_CMP_GT,
    SHMEM_CMP_LE,
    SHMEM_CMP_LT,
    SHMEM_CMP_GE
};

/*
 * Deprecated (but still valid) names
 */
#define _SHMEM_CMP_EQ                   SHMEM_CMP_EQ
#define _SHMEM_CMP_NE                   SHMEM_CMP_NE
#define _SHMEM_CMP_GT                   SHMEM_CMP_GT
#define _SHMEM_CMP_LE                   SHMEM_CMP_LE
#define _SHMEM_CMP_LT                   SHMEM_CMP_LT
#define _SHMEM_CMP_GE                   SHMEM_CMP_GE

#define _SHMEM_BARRIER_SYNC_SIZE        (1)
#define _SHMEM_BCAST_SYNC_SIZE          (1 + _SHMEM_BARRIER_SYNC_SIZE)
#define _SHMEM_COLLECT_SYNC_SIZE        (1 + _SHMEM_BCAST_SYNC_SIZE)
#define _SHMEM_REDUCE_SYNC_SIZE         (1 + _SHMEM_BCAST_SYNC_SIZE)
#define _SHMEM_ALLTOALL_SYNC_SIZE       (_SHMEM_BARRIER_SYNC_SIZE)
#define _SHMEM_ALLTOALLS_SYNC_SIZE      (_SHMEM_BARRIER_SYNC_SIZE)
#define _SHMEM_REDUCE_MIN_WRKDATA_SIZE  (1)
#define _SHMEM_SYNC_VALUE               (-1)

#define SHMEM_BARRIER_SYNC_SIZE        _SHMEM_BARRIER_SYNC_SIZE
#define SHMEM_BCAST_SYNC_SIZE          _SHMEM_BCAST_SYNC_SIZE
#define SHMEM_COLLECT_SYNC_SIZE        _SHMEM_COLLECT_SYNC_SIZE
#define SHMEM_REDUCE_SYNC_SIZE         _SHMEM_REDUCE_SYNC_SIZE
#define SHMEM_ALLTOALL_SYNC_SIZE       _SHMEM_ALLTOALL_SYNC_SIZE
#define SHMEM_ALLTOALLS_SYNC_SIZE      _SHMEM_ALLTOALLS_SYNC_SIZE
#define SHMEM_REDUCE_MIN_WRKDATA_SIZE  _SHMEM_REDUCE_MIN_WRKDATA_SIZE
#define SHMEM_SYNC_VALUE               _SHMEM_SYNC_VALUE
#define SHMEM_SYNC_SIZE                _SHMEM_COLLECT_SYNC_SIZE


/*
 * Initialization routines
 */
OSHMEM_DECLSPEC  void shmem_init(void);
OSHMEM_DECLSPEC  int shmem_init_thread(int requested, int *provided);
OSHMEM_DECLSPEC  void shmem_finalize(void);
OSHMEM_DECLSPEC  void shmem_global_exit(int status);

/*
 * Query routines
 */
OSHMEM_DECLSPEC  int shmem_n_pes(void);
OSHMEM_DECLSPEC  int shmem_my_pe(void);
OSHMEM_DECLSPEC  void shmem_query_thread(int *provided);

/*
 * Info routines
 */
OSHMEM_DECLSPEC void shmem_info_get_version(int *major, int *minor);
OSHMEM_DECLSPEC void shmem_info_get_name(char *name);

/*
 * Accessability routines
 */
OSHMEM_DECLSPEC int shmem_pe_accessible(int pe);
OSHMEM_DECLSPEC int shmem_addr_accessible(const void *addr, int pe);

/*
 * Symmetric heap routines
 */
OSHMEM_DECLSPEC  void* shmem_malloc(size_t size);
OSHMEM_DECLSPEC  void* shmem_calloc(size_t count, size_t size);
OSHMEM_DECLSPEC  void* shmem_align(size_t align, size_t size);
OSHMEM_DECLSPEC  void* shmem_realloc(void *ptr, size_t size);
OSHMEM_DECLSPEC  void shmem_free(void* ptr);

/*
 * Remote pointer operations
 */
OSHMEM_DECLSPEC  void *shmem_ptr(const void *ptr, int pe);

/*
 * Communication context operations
 */

typedef struct { int dummy; } * shmem_ctx_t;

#define SHMEM_CTX_DEFAULT oshmem_ctx_default

extern shmem_ctx_t oshmem_ctx_default;

OSHMEM_DECLSPEC int shmem_ctx_create(long options, shmem_ctx_t *ctx);
OSHMEM_DECLSPEC void shmem_ctx_destroy(shmem_ctx_t ctx);

/*
 * Elemental put routines
 */
OSHMEM_DECLSPEC  void shmem_ctx_char_p(shmem_ctx_t ctx, char* addr, char value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_short_p(shmem_ctx_t ctx, short* addr, short value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int_p(shmem_ctx_t ctx, int* addr, int value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_long_p(shmem_ctx_t ctx, long* addr, long value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_float_p(shmem_ctx_t ctx, float* addr, float value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_double_p(shmem_ctx_t ctx, double* addr, double value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_longlong_p(shmem_ctx_t ctx, long long* addr, long long value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_schar_p(shmem_ctx_t ctx, signed char* addr, signed char value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uchar_p(shmem_ctx_t ctx, unsigned char* addr, unsigned char value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ushort_p(shmem_ctx_t ctx, unsigned short* addr, unsigned short value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint_p(shmem_ctx_t ctx, unsigned int* addr, unsigned int value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ulong_p(shmem_ctx_t ctx, unsigned long* addr, unsigned long value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ulonglong_p(shmem_ctx_t ctx, unsigned long long* addr, unsigned long long value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_longdouble_p(shmem_ctx_t ctx, long double* addr, long double value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int8_p(shmem_ctx_t ctx, int8_t* addr, int8_t value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int16_p(shmem_ctx_t ctx, int16_t* addr, int16_t value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int32_p(shmem_ctx_t ctx, int32_t* addr, int32_t value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int64_p(shmem_ctx_t ctx, int64_t* addr, int64_t value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint8_p(shmem_ctx_t ctx, uint8_t* addr, uint8_t value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint16_p(shmem_ctx_t ctx, uint16_t* addr, uint16_t value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint32_p(shmem_ctx_t ctx, uint32_t* addr, uint32_t value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint64_p(shmem_ctx_t ctx, uint64_t* addr, uint64_t value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_size_p(shmem_ctx_t ctx, size_t* addr, size_t value, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ptrdiff_p(shmem_ctx_t ctx, ptrdiff_t* addr, ptrdiff_t value, int pe);

OSHMEM_DECLSPEC  void shmem_char_p(char* addr, char value, int pe);
OSHMEM_DECLSPEC  void shmem_short_p(short* addr, short value, int pe);
OSHMEM_DECLSPEC  void shmem_int_p(int* addr, int value, int pe);
OSHMEM_DECLSPEC  void shmem_long_p(long* addr, long value, int pe);
OSHMEM_DECLSPEC  void shmem_float_p(float* addr, float value, int pe);
OSHMEM_DECLSPEC  void shmem_double_p(double* addr, double value, int pe);
OSHMEM_DECLSPEC  void shmem_longlong_p(long long* addr, long long value, int pe);
OSHMEM_DECLSPEC  void shmem_schar_p(signed char* addr, signed char value, int pe);
OSHMEM_DECLSPEC  void shmem_uchar_p(unsigned char* addr, unsigned char value, int pe);
OSHMEM_DECLSPEC  void shmem_ushort_p(unsigned short* addr, unsigned short value, int pe);
OSHMEM_DECLSPEC  void shmem_uint_p(unsigned int* addr, unsigned int value, int pe);
OSHMEM_DECLSPEC  void shmem_ulong_p(unsigned long* addr, unsigned long value, int pe);
OSHMEM_DECLSPEC  void shmem_ulonglong_p(unsigned long long* addr, unsigned long long value, int pe);
OSHMEM_DECLSPEC  void shmem_longdouble_p(long double* addr, long double value, int pe);
OSHMEM_DECLSPEC  void shmem_int8_p(int8_t* addr, int8_t value, int pe);
OSHMEM_DECLSPEC  void shmem_int16_p(int16_t* addr, int16_t value, int pe);
OSHMEM_DECLSPEC  void shmem_int32_p(int32_t* addr, int32_t value, int pe);
OSHMEM_DECLSPEC  void shmem_int64_p(int64_t* addr, int64_t value, int pe);
OSHMEM_DECLSPEC  void shmem_uint8_p(uint8_t* addr, uint8_t value, int pe);
OSHMEM_DECLSPEC  void shmem_uint16_p(uint16_t* addr, uint16_t value, int pe);
OSHMEM_DECLSPEC  void shmem_uint32_p(uint32_t* addr, uint32_t value, int pe);
OSHMEM_DECLSPEC  void shmem_uint64_p(uint64_t* addr, uint64_t value, int pe);
OSHMEM_DECLSPEC  void shmem_size_p(size_t* addr, size_t value, int pe);
OSHMEM_DECLSPEC  void shmem_ptrdiff_p(ptrdiff_t* addr, ptrdiff_t value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_p(...)                                                 \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                     \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)), \
                char*:        shmem_ctx_char_p,                      \
                short*:       shmem_ctx_short_p,                     \
                int*:         shmem_ctx_int_p,                       \
                long*:        shmem_ctx_long_p,                      \
                long long*:   shmem_ctx_longlong_p,                  \
                signed char*:        shmem_ctx_schar_p,              \
                unsigned char*:      shmem_ctx_uchar_p,              \
                unsigned short*:     shmem_ctx_ushort_p,             \
                unsigned int*:       shmem_ctx_uint_p,               \
                unsigned long*:      shmem_ctx_ulong_p,              \
                unsigned long long*: shmem_ctx_ulonglong_p,          \
                float*:       shmem_ctx_float_p,                     \
                double*:      shmem_ctx_double_p,                    \
                long double*: shmem_ctx_longdouble_p,                \
                default:      __oshmem_datatype_ignore),             \
            char*:        shmem_char_p,                              \
            short*:       shmem_short_p,                             \
            int*:         shmem_int_p,                               \
            long*:        shmem_long_p,                              \
            long long*:   shmem_longlong_p,                          \
            signed char*:        shmem_schar_p,                      \
            unsigned char*:      shmem_uchar_p,                      \
            unsigned short*:     shmem_ushort_p,                     \
            unsigned int*:       shmem_uint_p,                       \
            unsigned long*:      shmem_ulong_p,                      \
            unsigned long long*: shmem_ulonglong_p,                  \
            float*:       shmem_float_p,                             \
            double*:      shmem_double_p,                            \
            long double*: shmem_longdouble_p)(__VA_ARGS__)
#endif

/*
 * Block data put routines
 */
OSHMEM_DECLSPEC  void shmem_ctx_char_put(shmem_ctx_t ctx, char *target, const char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_short_put(shmem_ctx_t ctx, short *target, const short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int_put(shmem_ctx_t ctx, int *target, const int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_long_put(shmem_ctx_t ctx, long *target, const long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_float_put(shmem_ctx_t ctx, float *target, const float *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_double_put(shmem_ctx_t ctx, double *target, const double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_longlong_put(shmem_ctx_t ctx, long long *target, const long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_schar_put(shmem_ctx_t ctx, signed char *target, const signed char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uchar_put(shmem_ctx_t ctx, unsigned char *target, const unsigned char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ushort_put(shmem_ctx_t ctx, unsigned short *target, const unsigned short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint_put(shmem_ctx_t ctx, unsigned int *target, const unsigned int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ulong_put(shmem_ctx_t ctx, unsigned long *target, const unsigned long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ulonglong_put(shmem_ctx_t ctx, unsigned long long *target, const unsigned long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_longdouble_put(shmem_ctx_t ctx, long double *target, const long double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int8_put(shmem_ctx_t ctx, int8_t *target, const int8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int16_put(shmem_ctx_t ctx, int16_t *target, const int16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int32_put(shmem_ctx_t ctx, int32_t *target, const int32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int64_put(shmem_ctx_t ctx, int64_t *target, const int64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint8_put(shmem_ctx_t ctx, uint8_t *target, const uint8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint16_put(shmem_ctx_t ctx, uint16_t *target, const uint16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint32_put(shmem_ctx_t ctx, uint32_t *target, const uint32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint64_put(shmem_ctx_t ctx, uint64_t *target, const uint64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_size_put(shmem_ctx_t ctx, size_t *target, const size_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ptrdiff_put(shmem_ctx_t ctx, ptrdiff_t *target, const ptrdiff_t *source, size_t len, int pe);

OSHMEM_DECLSPEC  void shmem_char_put(char *target, const char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_short_put(short *target, const short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int_put(int *target, const int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_long_put(long *target, const long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_float_put(float *target, const float *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_double_put(double *target, const double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_longlong_put(long long *target, const long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_schar_put(signed char *target, const signed char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uchar_put(unsigned char *target, const unsigned char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ushort_put(unsigned short *target, const unsigned short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint_put(unsigned int *target, const unsigned int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ulong_put(unsigned long *target, const unsigned long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ulonglong_put(unsigned long long *target, const unsigned long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_longdouble_put(long double *target, const long double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int8_put(int8_t *target, const int8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int16_put(int16_t *target, const int16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int32_put(int32_t *target, const int32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int64_put(int64_t *target, const int64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint8_put(uint8_t *target, const uint8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint16_put(uint16_t *target, const uint16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint32_put(uint32_t *target, const uint32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint64_put(uint64_t *target, const uint64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_size_put(size_t *target, const size_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ptrdiff_put(ptrdiff_t *target, const ptrdiff_t *source, size_t len, int pe);
#if OSHMEM_HAVE_C11
#define shmem_put(...)                                              \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                    \
            shmem_ctx_t: _Generic(&*(__OSHMEM_VAR_ARG2(__VA_ARGS__)),\
                char*:        shmem_ctx_char_put,                   \
                short*:       shmem_ctx_short_put,                  \
                int*:         shmem_ctx_int_put,                    \
                long*:        shmem_ctx_long_put,                   \
                long long*:   shmem_ctx_longlong_put,               \
                signed char*:        shmem_ctx_schar_put,           \
                unsigned char*:      shmem_ctx_uchar_put,           \
                unsigned short*:     shmem_ctx_ushort_put,          \
                unsigned int*:       shmem_ctx_uint_put,            \
                unsigned long*:      shmem_ctx_ulong_put,           \
                unsigned long long*: shmem_ctx_ulonglong_put,       \
                float*:       shmem_ctx_float_put,                  \
                double*:      shmem_ctx_double_put,                 \
                long double*: shmem_ctx_longdouble_put,             \
                default:      __oshmem_datatype_ignore),            \
            char*:        shmem_char_put,                           \
            short*:       shmem_short_put,                          \
            int*:         shmem_int_put,                            \
            long*:        shmem_long_put,                           \
            long long*:   shmem_longlong_put,                       \
            signed char*:        shmem_schar_put,                   \
            unsigned char*:      shmem_uchar_put,                   \
            unsigned short*:     shmem_ushort_put,                  \
            unsigned int*:       shmem_uint_put,                    \
            unsigned long*:      shmem_ulong_put,                   \
            unsigned long long*: shmem_ulonglong_put,               \
            float*:       shmem_float_put,                          \
            double*:      shmem_double_put,                         \
            long double*: shmem_longdouble_put)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC  void shmem_ctx_put8(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_put16(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_put32(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_put64(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_put128(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_putmem(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);

OSHMEM_DECLSPEC  void shmem_put8(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_put16(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_put32(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_put64(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_put128(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_putmem(void *target, const void *source, size_t len, int pe);


/*
 * Strided put routines
 */
OSHMEM_DECLSPEC void shmem_ctx_char_iput(shmem_ctx_t ctx, char* target, const char* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_short_iput(shmem_ctx_t ctx, short* target, const short* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int_iput(shmem_ctx_t ctx, int* target, const int* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_long_iput(shmem_ctx_t ctx, long* target, const long* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_float_iput(shmem_ctx_t ctx, float* target, const float* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_double_iput(shmem_ctx_t ctx, double* target, const double* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_longlong_iput(shmem_ctx_t ctx, long long* target, const long long* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_schar_iput(shmem_ctx_t ctx, signed char* target, const signed char* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uchar_iput(shmem_ctx_t ctx, unsigned char* target, const unsigned char* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ushort_iput(shmem_ctx_t ctx, unsigned short* target, const unsigned short* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint_iput(shmem_ctx_t ctx, unsigned int* target, const unsigned int* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulong_iput(shmem_ctx_t ctx, unsigned long* target, const unsigned long* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulonglong_iput(shmem_ctx_t ctx, unsigned long long* target, const unsigned long long* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_longdouble_iput(shmem_ctx_t ctx, long double* target, const long double* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int8_iput(shmem_ctx_t ctx, int8_t* target, const int8_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int16_iput(shmem_ctx_t ctx, int16_t* target, const int16_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int32_iput(shmem_ctx_t ctx, int32_t* target, const int32_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int64_iput(shmem_ctx_t ctx, int64_t* target, const int64_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint8_iput(shmem_ctx_t ctx, uint8_t* target, const uint8_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint16_iput(shmem_ctx_t ctx, uint16_t* target, const uint16_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint32_iput(shmem_ctx_t ctx, uint32_t* target, const uint32_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint64_iput(shmem_ctx_t ctx, uint64_t* target, const uint64_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_size_iput(shmem_ctx_t ctx, size_t* target, const size_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ptrdiff_iput(shmem_ctx_t ctx, ptrdiff_t* target, const ptrdiff_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);

OSHMEM_DECLSPEC void shmem_char_iput(char* target, const char* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_short_iput(short* target, const short* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_int_iput(int* target, const int* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_long_iput(long* target, const long* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_float_iput(float* target, const float* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_double_iput(double* target, const double* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_longlong_iput(long long* target, const long long* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_schar_iput(signed char* target, const signed char* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uchar_iput(unsigned char* target, const unsigned char* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ushort_iput(unsigned short* target, const unsigned short* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uint_iput(unsigned int* target, const unsigned int* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ulong_iput(unsigned long* target, const unsigned long* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ulonglong_iput(unsigned long long* target, const unsigned long long* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_longdouble_iput(long double* target, const long double* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_int8_iput(int8_t* target, const int8_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_int16_iput(int16_t* target, const int16_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_int32_iput(int32_t* target, const int32_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_int64_iput(int64_t* target, const int64_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uint8_iput(uint8_t* target, const uint8_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uint16_iput(uint16_t* target, const uint16_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uint32_iput(uint32_t* target, const uint32_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uint64_iput(uint64_t* target, const uint64_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_size_iput(size_t* target, const size_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ptrdiff_iput(ptrdiff_t* target, const ptrdiff_t* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
#if OSHMEM_HAVE_C11
#define shmem_iput(...)                                              \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                     \
            shmem_ctx_t:  _Generic(&*(__OSHMEM_VAR_ARG2(__VA_ARGS__)),\
                char*:        shmem_ctx_char_iput,                   \
                short*:       shmem_ctx_short_iput,                  \
                int*:         shmem_ctx_int_iput,                    \
                long*:        shmem_ctx_long_iput,                   \
                long long*:   shmem_ctx_longlong_iput,               \
                signed char*:        shmem_ctx_schar_iput,           \
                unsigned char*:      shmem_ctx_uchar_iput,           \
                unsigned short*:     shmem_ctx_ushort_iput,          \
                unsigned int*:       shmem_ctx_uint_iput,            \
                unsigned long*:      shmem_ctx_ulong_iput,           \
                unsigned long long*: shmem_ctx_ulonglong_iput,       \
                float*:       shmem_ctx_float_iput,                  \
                double*:      shmem_ctx_double_iput,                 \
                long double*: shmem_ctx_longdouble_iput,             \
                default:      __oshmem_datatype_ignore),             \
            char*:        shmem_char_iput,                           \
            short*:       shmem_short_iput,                          \
            int*:         shmem_int_iput,                            \
            long*:        shmem_long_iput,                           \
            long long*:   shmem_longlong_iput,                       \
            signed char*:        shmem_schar_iput,                   \
            unsigned char*:      shmem_uchar_iput,                   \
            unsigned short*:     shmem_ushort_iput,                  \
            unsigned int*:       shmem_uint_iput,                    \
            unsigned long*:      shmem_ulong_iput,                   \
            unsigned long long*: shmem_ulonglong_iput,               \
            float*:       shmem_float_iput,                          \
            double*:      shmem_double_iput,                         \
            long double*: shmem_longdouble_iput)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC void shmem_ctx_iput8(shmem_ctx_t ctx, void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_iput16(shmem_ctx_t ctx, void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_iput32(shmem_ctx_t ctx, void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_iput64(shmem_ctx_t ctx, void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_iput128(shmem_ctx_t ctx, void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);

OSHMEM_DECLSPEC void shmem_iput8(void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_iput16(void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_iput32(void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_iput64(void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_iput128(void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);

/*
 * Nonblocking put routines
 */
OSHMEM_DECLSPEC  void shmem_ctx_char_put_nbi(shmem_ctx_t ctx, char *target, const char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_short_put_nbi(shmem_ctx_t ctx, short *target, const short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int_put_nbi(shmem_ctx_t ctx, int *target, const int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_long_put_nbi(shmem_ctx_t ctx, long *target, const long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_float_put_nbi(shmem_ctx_t ctx, float *target, const float *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_double_put_nbi(shmem_ctx_t ctx, double *target, const double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_longlong_put_nbi(shmem_ctx_t ctx, long long *target, const long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_schar_put_nbi(shmem_ctx_t ctx, signed char *target, const signed char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uchar_put_nbi(shmem_ctx_t ctx, unsigned char *target, const unsigned char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ushort_put_nbi(shmem_ctx_t ctx, unsigned short *target, const unsigned short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint_put_nbi(shmem_ctx_t ctx, unsigned int *target, const unsigned int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ulong_put_nbi(shmem_ctx_t ctx, unsigned long *target, const unsigned long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ulonglong_put_nbi(shmem_ctx_t ctx, unsigned long long *target, const unsigned long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_longdouble_put_nbi(shmem_ctx_t ctx, long double *target, const long double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int8_put_nbi(shmem_ctx_t ctx, int8_t *target, const int8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int16_put_nbi(shmem_ctx_t ctx, int16_t *target, const int16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int32_put_nbi(shmem_ctx_t ctx, int32_t *target, const int32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int64_put_nbi(shmem_ctx_t ctx, int64_t *target, const int64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint8_put_nbi(shmem_ctx_t ctx, uint8_t *target, const uint8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint16_put_nbi(shmem_ctx_t ctx, uint16_t *target, const uint16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint32_put_nbi(shmem_ctx_t ctx, uint32_t *target, const uint32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint64_put_nbi(shmem_ctx_t ctx, uint64_t *target, const uint64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_size_put_nbi(shmem_ctx_t ctx, size_t *target, const size_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ptrdiff_put_nbi(shmem_ctx_t ctx, ptrdiff_t *target, const ptrdiff_t *source, size_t len, int pe);

OSHMEM_DECLSPEC  void shmem_char_put_nbi(char *target, const char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_short_put_nbi(short *target, const short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int_put_nbi(int *target, const int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_long_put_nbi(long *target, const long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_float_put_nbi(float *target, const float *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_double_put_nbi(double *target, const double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_longlong_put_nbi(long long *target, const long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_schar_put_nbi(signed char *target, const signed char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uchar_put_nbi(unsigned char *target, const unsigned char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ushort_put_nbi(unsigned short *target, const unsigned short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint_put_nbi(unsigned int *target, const unsigned int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ulong_put_nbi(unsigned long *target, const unsigned long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ulonglong_put_nbi(unsigned long long *target, const unsigned long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_longdouble_put_nbi(long double *target, const long double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int8_put_nbi(int8_t *target, const int8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int16_put_nbi(int16_t *target, const int16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int32_put_nbi(int32_t *target, const int32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int64_put_nbi(int64_t *target, const int64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint8_put_nbi(uint8_t *target, const uint8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint16_put_nbi(uint16_t *target, const uint16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint32_put_nbi(uint32_t *target, const uint32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint64_put_nbi(uint64_t *target, const uint64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_size_put_nbi(size_t *target, const size_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ptrdiff_put_nbi(ptrdiff_t *target, const ptrdiff_t *source, size_t len, int pe);
#if OSHMEM_HAVE_C11
#define shmem_put_nbi(...)                                              \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                        \
            shmem_ctx_t: _Generic(&*(__OSHMEM_VAR_ARG2(__VA_ARGS__)),   \
                char*:        shmem_ctx_char_put_nbi,                   \
                short*:       shmem_ctx_short_put_nbi,                  \
                int*:         shmem_ctx_int_put_nbi,                    \
                long*:        shmem_ctx_long_put_nbi,                   \
                long long*:   shmem_ctx_longlong_put_nbi,               \
                signed char*:        shmem_ctx_schar_put_nbi,           \
                unsigned char*:      shmem_ctx_uchar_put_nbi,           \
                unsigned short*:     shmem_ctx_ushort_put_nbi,          \
                unsigned int*:       shmem_ctx_uint_put_nbi,            \
                unsigned long*:      shmem_ctx_ulong_put_nbi,           \
                unsigned long long*: shmem_ctx_ulonglong_put_nbi,       \
                float*:       shmem_ctx_float_put_nbi,                  \
                double*:      shmem_ctx_double_put_nbi,                 \
                long double*: shmem_ctx_longdouble_put_nbi,             \
                default:      __oshmem_datatype_ignore),                \
            char*:        shmem_char_put_nbi,                           \
            short*:       shmem_short_put_nbi,                          \
            int*:         shmem_int_put_nbi,                            \
            long*:        shmem_long_put_nbi,                           \
            long long*:   shmem_longlong_put_nbi,                       \
            signed char*:        shmem_schar_put_nbi,                   \
            unsigned char*:      shmem_uchar_put_nbi,                   \
            unsigned short*:     shmem_ushort_put_nbi,                  \
            unsigned int*:       shmem_uint_put_nbi,                    \
            unsigned long*:      shmem_ulong_put_nbi,                   \
            unsigned long long*: shmem_ulonglong_put_nbi,               \
            float*:       shmem_float_put_nbi,                          \
            double*:      shmem_double_put_nbi,                         \
            long double*: shmem_longdouble_put_nbi)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC  void shmem_ctx_put8_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_put16_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_put32_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_put64_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_put128_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_putmem_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);

OSHMEM_DECLSPEC  void shmem_put8_nbi(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_put16_nbi(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_put32_nbi(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_put64_nbi(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_put128_nbi(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_putmem_nbi(void *target, const void *source, size_t len, int pe);

/*
 * Elemental get routines
 */
OSHMEM_DECLSPEC  char shmem_ctx_char_g(shmem_ctx_t ctx, const char* addr, int pe);
OSHMEM_DECLSPEC  short shmem_ctx_short_g(shmem_ctx_t ctx, const short* addr, int pe);
OSHMEM_DECLSPEC  int shmem_ctx_int_g(shmem_ctx_t ctx, const int* addr, int pe);
OSHMEM_DECLSPEC  long shmem_ctx_long_g(shmem_ctx_t ctx, const long* addr, int pe);
OSHMEM_DECLSPEC  float shmem_ctx_float_g(shmem_ctx_t ctx, const float* addr, int pe);
OSHMEM_DECLSPEC  double shmem_ctx_double_g(shmem_ctx_t ctx, const double* addr, int pe);
OSHMEM_DECLSPEC  long long shmem_ctx_longlong_g(shmem_ctx_t ctx, const long long* addr, int pe);
OSHMEM_DECLSPEC  long double shmem_ctx_longdouble_g(shmem_ctx_t ctx, const long double* addr, int pe);
OSHMEM_DECLSPEC  signed char shmem_ctx_schar_g(shmem_ctx_t ctx, const signed char* addr, int pe);
OSHMEM_DECLSPEC  unsigned char shmem_ctx_uchar_g(shmem_ctx_t ctx, const unsigned char* addr, int pe);
OSHMEM_DECLSPEC  unsigned short shmem_ctx_ushort_g(shmem_ctx_t ctx, const unsigned short* addr, int pe);
OSHMEM_DECLSPEC  unsigned int shmem_ctx_uint_g(shmem_ctx_t ctx, const unsigned int* addr, int pe);
OSHMEM_DECLSPEC  unsigned long shmem_ctx_ulong_g(shmem_ctx_t ctx, const unsigned long* addr, int pe);
OSHMEM_DECLSPEC  unsigned long long shmem_ctx_ulonglong_g(shmem_ctx_t ctx, const unsigned long long* addr, int pe);
OSHMEM_DECLSPEC  int8_t shmem_ctx_int8_g(shmem_ctx_t ctx, const int8_t* addr, int pe);
OSHMEM_DECLSPEC  int16_t shmem_ctx_int16_g(shmem_ctx_t ctx, const int16_t* addr, int pe);
OSHMEM_DECLSPEC  int32_t shmem_ctx_int32_g(shmem_ctx_t ctx, const int32_t* addr, int pe);
OSHMEM_DECLSPEC  int64_t shmem_ctx_int64_g(shmem_ctx_t ctx, const int64_t* addr, int pe);
OSHMEM_DECLSPEC  uint8_t shmem_ctx_uint8_g(shmem_ctx_t ctx, const uint8_t* addr, int pe);
OSHMEM_DECLSPEC  uint16_t shmem_ctx_uint16_g(shmem_ctx_t ctx, const uint16_t* addr, int pe);
OSHMEM_DECLSPEC  uint32_t shmem_ctx_uint32_g(shmem_ctx_t ctx, const uint32_t* addr, int pe);
OSHMEM_DECLSPEC  uint64_t shmem_ctx_uint64_g(shmem_ctx_t ctx, const uint64_t* addr, int pe);
OSHMEM_DECLSPEC  size_t shmem_ctx_size_g(shmem_ctx_t ctx, const size_t* addr, int pe);
OSHMEM_DECLSPEC  ptrdiff_t shmem_ctx_ptrdiff_g(shmem_ctx_t ctx, const ptrdiff_t* addr, int pe);

OSHMEM_DECLSPEC  char shmem_char_g(const char* addr, int pe);
OSHMEM_DECLSPEC  short shmem_short_g(const short* addr, int pe);
OSHMEM_DECLSPEC  int shmem_int_g(const int* addr, int pe);
OSHMEM_DECLSPEC  long shmem_long_g(const long* addr, int pe);
OSHMEM_DECLSPEC  float shmem_float_g(const float* addr, int pe);
OSHMEM_DECLSPEC  double shmem_double_g(const double* addr, int pe);
OSHMEM_DECLSPEC  long long shmem_longlong_g(const long long* addr, int pe);
OSHMEM_DECLSPEC  long double shmem_longdouble_g(const long double* addr, int pe);
OSHMEM_DECLSPEC  signed char shmem_schar_g(const signed char* addr, int pe);
OSHMEM_DECLSPEC  unsigned char shmem_uchar_g(const unsigned char* addr, int pe);
OSHMEM_DECLSPEC  unsigned short shmem_ushort_g(const unsigned short* addr, int pe);
OSHMEM_DECLSPEC  unsigned int shmem_uint_g(const unsigned int* addr, int pe);
OSHMEM_DECLSPEC  unsigned long shmem_ulong_g(const unsigned long* addr, int pe);
OSHMEM_DECLSPEC  unsigned long long shmem_ulonglong_g(const unsigned long long* addr, int pe);
OSHMEM_DECLSPEC  int8_t shmem_int8_g(const int8_t* addr, int pe);
OSHMEM_DECLSPEC  int16_t shmem_int16_g(const int16_t* addr, int pe);
OSHMEM_DECLSPEC  int32_t shmem_int32_g(const int32_t* addr, int pe);
OSHMEM_DECLSPEC  int64_t shmem_int64_g(const int64_t* addr, int pe);
OSHMEM_DECLSPEC  uint8_t shmem_uint8_g(const uint8_t* addr, int pe);
OSHMEM_DECLSPEC  uint16_t shmem_uint16_g(const uint16_t* addr, int pe);
OSHMEM_DECLSPEC  uint32_t shmem_uint32_g(const uint32_t* addr, int pe);
OSHMEM_DECLSPEC  uint64_t shmem_uint64_g(const uint64_t* addr, int pe);
OSHMEM_DECLSPEC  size_t shmem_size_g(const size_t* addr, int pe);
OSHMEM_DECLSPEC  ptrdiff_t shmem_ptrdiff_g(const ptrdiff_t* addr, int pe);
#if OSHMEM_HAVE_C11
#define shmem_g(...)                                                \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                    \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)),\
                char*:        shmem_ctx_char_g,                     \
                short*:       shmem_ctx_short_g,                    \
                int*:         shmem_ctx_int_g,                      \
                long*:        shmem_ctx_long_g,                     \
                long long*:   shmem_ctx_longlong_g,                 \
                signed char*:        shmem_ctx_schar_g,             \
                unsigned char*:      shmem_ctx_uchar_g,             \
                unsigned short*:     shmem_ctx_ushort_g,            \
                unsigned int*:       shmem_ctx_uint_g,              \
                unsigned long*:      shmem_ctx_ulong_g,             \
                unsigned long long*: shmem_ctx_ulonglong_g,         \
                float*:       shmem_ctx_float_g,                    \
                double*:      shmem_ctx_double_g,                   \
                long double*: shmem_ctx_longdouble_g,               \
                default:      __oshmem_datatype_ignore),            \
            char*:        shmem_char_g,                             \
            short*:       shmem_short_g,                            \
            int*:         shmem_int_g,                              \
            long*:        shmem_long_g,                             \
            long long*:   shmem_longlong_g,                         \
            signed char*:        shmem_schar_g,                     \
            unsigned char*:      shmem_uchar_g,                     \
            unsigned short*:     shmem_ushort_g,                    \
            unsigned int*:       shmem_uint_g,                      \
            unsigned long*:      shmem_ulong_g,                     \
            unsigned long long*: shmem_ulonglong_g,                 \
            float*:       shmem_float_g,                            \
            double*:      shmem_double_g,                           \
            long double*: shmem_longdouble_g)(__VA_ARGS__)
#endif

/*
 * Block data get routines
 */
OSHMEM_DECLSPEC  void shmem_ctx_char_get(shmem_ctx_t ctx, char *target, const char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_short_get(shmem_ctx_t ctx, short *target, const short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int_get(shmem_ctx_t ctx, int *target, const int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_long_get(shmem_ctx_t ctx, long *target, const long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_float_get(shmem_ctx_t ctx, float *target, const float *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_double_get(shmem_ctx_t ctx, double *target, const double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_longlong_get(shmem_ctx_t ctx, long long *target, const long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_schar_get(shmem_ctx_t ctx, signed char *target, const signed char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uchar_get(shmem_ctx_t ctx, unsigned char *target, const unsigned char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ushort_get(shmem_ctx_t ctx, unsigned short *target, const unsigned short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint_get(shmem_ctx_t ctx, unsigned int *target, const unsigned int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ulong_get(shmem_ctx_t ctx, unsigned long *target, const unsigned long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ulonglong_get(shmem_ctx_t ctx, unsigned long long *target, const unsigned long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_longdouble_get(shmem_ctx_t ctx, long double *target, const long double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int8_get(shmem_ctx_t ctx, int8_t *target, const int8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int16_get(shmem_ctx_t ctx, int16_t *target, const int16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int32_get(shmem_ctx_t ctx, int32_t *target, const int32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int64_get(shmem_ctx_t ctx, int64_t *target, const int64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint8_get(shmem_ctx_t ctx, uint8_t *target, const uint8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint16_get(shmem_ctx_t ctx, uint16_t *target, const uint16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint32_get(shmem_ctx_t ctx, uint32_t *target, const uint32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint64_get(shmem_ctx_t ctx, uint64_t *target, const uint64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_size_get(shmem_ctx_t ctx, size_t *target, const size_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ptrdiff_get(shmem_ctx_t ctx, ptrdiff_t *target, const ptrdiff_t *source, size_t len, int pe);

OSHMEM_DECLSPEC  void shmem_char_get(char *target, const char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_short_get(short *target, const short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int_get(int *target, const int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_long_get(long *target, const long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_float_get(float *target, const float *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_double_get(double *target, const double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_longlong_get(long long *target, const long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_schar_get(signed char *target, const signed char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uchar_get(unsigned char *target, const unsigned char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ushort_get(unsigned short *target, const unsigned short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint_get(unsigned int *target, const unsigned int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ulong_get(unsigned long *target, const unsigned long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ulonglong_get(unsigned long long *target, const unsigned long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_longdouble_get(long double *target, const long double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int8_get(int8_t *target, const int8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int16_get(int16_t *target, const int16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int32_get(int32_t *target, const int32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int64_get(int64_t *target, const int64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint8_get(uint8_t *target, const uint8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint16_get(uint16_t *target, const uint16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint32_get(uint32_t *target, const uint32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint64_get(uint64_t *target, const uint64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_size_get(size_t *target, const size_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ptrdiff_get(ptrdiff_t *target, const ptrdiff_t *source, size_t len, int pe);
#if OSHMEM_HAVE_C11
#define shmem_get(...)                                              \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                    \
            shmem_ctx_t: _Generic(&*(__OSHMEM_VAR_ARG2(__VA_ARGS__)),\
                char*:        shmem_ctx_char_get,                   \
                short*:       shmem_ctx_short_get,                  \
                int*:         shmem_ctx_int_get,                    \
                long*:        shmem_ctx_long_get,                   \
                long long*:   shmem_ctx_longlong_get,               \
                signed char*:        shmem_ctx_schar_get,           \
                unsigned char*:      shmem_ctx_uchar_get,           \
                unsigned short*:     shmem_ctx_ushort_get,          \
                unsigned int*:       shmem_ctx_uint_get,            \
                unsigned long*:      shmem_ctx_ulong_get,           \
                unsigned long long*: shmem_ctx_ulonglong_get,       \
                float*:       shmem_ctx_float_get,                  \
                double*:      shmem_ctx_double_get,                 \
                long double*: shmem_ctx_longdouble_get,             \
                default:      __oshmem_datatype_ignore),            \
            char*:        shmem_char_get,                           \
            short*:       shmem_short_get,                          \
            int*:         shmem_int_get,                            \
            long*:        shmem_long_get,                           \
            long long*:   shmem_longlong_get,                       \
            signed char*:        shmem_schar_get,                   \
            unsigned char*:      shmem_uchar_get,                   \
            unsigned short*:     shmem_ushort_get,                  \
            unsigned int*:       shmem_uint_get,                    \
            unsigned long*:      shmem_ulong_get,                   \
            unsigned long long*: shmem_ulonglong_get,               \
            float*:       shmem_float_get,                          \
            double*:      shmem_double_get,                         \
            long double*: shmem_longdouble_get)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC  void shmem_ctx_get8(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_get16(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_get32(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_get64(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_get128(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_getmem(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);

OSHMEM_DECLSPEC  void shmem_get8(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_get16(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_get32(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_get64(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_get128(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_getmem(void *target, const void *source, size_t len, int pe);

/*
 * Strided get routines
 */
OSHMEM_DECLSPEC void shmem_ctx_char_iget(shmem_ctx_t ctx, char* target, const char* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_short_iget(shmem_ctx_t ctx, short* target, const short* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int_iget(shmem_ctx_t ctx, int* target, const int* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_long_iget(shmem_ctx_t ctx, long* target, const long* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_longlong_iget(shmem_ctx_t ctx, long long* target, const long long* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_schar_iget(shmem_ctx_t ctx, signed char* target, const signed char* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uchar_iget(shmem_ctx_t ctx, unsigned char* target, const unsigned char* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ushort_iget(shmem_ctx_t ctx, unsigned short* target, const unsigned short* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint_iget(shmem_ctx_t ctx, unsigned int* target, const unsigned int* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulong_iget(shmem_ctx_t ctx, unsigned long* target, const unsigned long* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulonglong_iget(shmem_ctx_t ctx, unsigned long long* target, const unsigned long long* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_float_iget(shmem_ctx_t ctx, float* target, const float* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_double_iget(shmem_ctx_t ctx, double* target, const double* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_longdouble_iget(shmem_ctx_t ctx, long double* target, const long double* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int8_iget(shmem_ctx_t ctx, int8_t* target, const int8_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int16_iget(shmem_ctx_t ctx, int16_t* target, const int16_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int32_iget(shmem_ctx_t ctx, int32_t* target, const int32_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int64_iget(shmem_ctx_t ctx, int64_t* target, const int64_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint8_iget(shmem_ctx_t ctx, uint8_t* target, const uint8_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint16_iget(shmem_ctx_t ctx, uint16_t* target, const uint16_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint32_iget(shmem_ctx_t ctx, uint32_t* target, const uint32_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint64_iget(shmem_ctx_t ctx, uint64_t* target, const uint64_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_size_iget(shmem_ctx_t ctx, size_t* target, const size_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ptrdiff_iget(shmem_ctx_t ctx, ptrdiff_t* target, const ptrdiff_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);

OSHMEM_DECLSPEC void shmem_char_iget(char* target, const char* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_short_iget(short* target, const short* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_int_iget(int* target, const int* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_float_iget(float* target, const float* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_double_iget(double* target, const double* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_longlong_iget(long long* target, const long long* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_longdouble_iget(long double* target, const long double* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_long_iget(long* target, const long* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_schar_iget(signed char* target, const signed char* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uchar_iget(unsigned char* target, const unsigned char* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ushort_iget(unsigned short* target, const unsigned short* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uint_iget(unsigned int* target, const unsigned int* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ulong_iget(unsigned long* target, const unsigned long* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ulonglong_iget(unsigned long long* target, const unsigned long long* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_int8_iget(int8_t* target, const int8_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_int16_iget(int16_t* target, const int16_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_int32_iget(int32_t* target, const int32_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_int64_iget(int64_t* target, const int64_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uint8_iget(uint8_t* target, const uint8_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uint16_iget(uint16_t* target, const uint16_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uint32_iget(uint32_t* target, const uint32_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_uint64_iget(uint64_t* target, const uint64_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_size_iget(size_t* target, const size_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ptrdiff_iget(ptrdiff_t* target, const ptrdiff_t* source, ptrdiff_t tst, ptrdiff_t sst, size_t len, int pe);
#if OSHMEM_HAVE_C11
#define shmem_iget(...)                                              \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                     \
            shmem_ctx_t: _Generic(&*(__OSHMEM_VAR_ARG2(__VA_ARGS__)), \
                char*:        shmem_ctx_char_iget,                   \
                short*:       shmem_ctx_short_iget,                  \
                int*:         shmem_ctx_int_iget,                    \
                long*:        shmem_ctx_long_iget,                   \
                long long*:   shmem_ctx_longlong_iget,               \
                signed char*:        shmem_ctx_schar_iget,           \
                unsigned char*:      shmem_ctx_uchar_iget,           \
                unsigned short*:     shmem_ctx_ushort_iget,          \
                unsigned int*:       shmem_ctx_uint_iget,            \
                unsigned long*:      shmem_ctx_ulong_iget,           \
                unsigned long long*: shmem_ctx_ulonglong_iget,       \
                float*:       shmem_ctx_float_iget,                  \
                double*:      shmem_ctx_double_iget,                 \
                long double*: shmem_ctx_longdouble_iget,             \
                default:      __oshmem_datatype_ignore),             \
            char*:        shmem_char_iget,                           \
            short*:       shmem_short_iget,                          \
            int*:         shmem_int_iget,                            \
            long*:        shmem_long_iget,                           \
            long long*:   shmem_longlong_iget,                       \
            signed char*:        shmem_schar_iget,                   \
            unsigned char*:      shmem_uchar_iget,                   \
            unsigned short*:     shmem_ushort_iget,                  \
            unsigned int*:       shmem_uint_iget,                    \
            unsigned long*:      shmem_ulong_iget,                   \
            unsigned long long*: shmem_ulonglong_iget,               \
            float*:       shmem_float_iget,                          \
            double*:      shmem_double_iget,                         \
            long double*: shmem_longdouble_iget)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC void shmem_ctx_iget8(shmem_ctx_t ctx, void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_iget16(shmem_ctx_t ctx, void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_iget32(shmem_ctx_t ctx, void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_iget64(shmem_ctx_t ctx, void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_ctx_iget128(shmem_ctx_t ctx, void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);

OSHMEM_DECLSPEC void shmem_iget8(void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_iget16(void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_iget32(void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_iget64(void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);
OSHMEM_DECLSPEC void shmem_iget128(void* target, const void* source, ptrdiff_t tst, ptrdiff_t sst,size_t len, int pe);

/*
 * Nonblocking data get routines
 */
OSHMEM_DECLSPEC  void shmem_ctx_char_get_nbi(shmem_ctx_t ctx, char *target, const char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_short_get_nbi(shmem_ctx_t ctx, short *target, const short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int_get_nbi(shmem_ctx_t ctx, int *target, const int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_long_get_nbi(shmem_ctx_t ctx, long *target, const long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_longlong_get_nbi(shmem_ctx_t ctx, long long *target, const long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_schar_get_nbi(shmem_ctx_t ctx, signed char *target, const signed char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uchar_get_nbi(shmem_ctx_t ctx, unsigned char *target, const unsigned char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ushort_get_nbi(shmem_ctx_t ctx, unsigned short *target, const unsigned short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint_get_nbi(shmem_ctx_t ctx, unsigned int *target, const unsigned int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ulong_get_nbi(shmem_ctx_t ctx, unsigned long *target, const unsigned long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ulonglong_get_nbi(shmem_ctx_t ctx, unsigned long long *target, const unsigned long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_float_get_nbi(shmem_ctx_t ctx, float *target, const float *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_double_get_nbi(shmem_ctx_t ctx, double *target, const double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_longdouble_get_nbi(shmem_ctx_t ctx, long double *target, const long double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int8_get_nbi(shmem_ctx_t ctx, int8_t *target, const int8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int16_get_nbi(shmem_ctx_t ctx, int16_t *target, const int16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int32_get_nbi(shmem_ctx_t ctx, int32_t *target, const int32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_int64_get_nbi(shmem_ctx_t ctx, int64_t *target, const int64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint8_get_nbi(shmem_ctx_t ctx, uint8_t *target, const uint8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint16_get_nbi(shmem_ctx_t ctx, uint16_t *target, const uint16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint32_get_nbi(shmem_ctx_t ctx, uint32_t *target, const uint32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_uint64_get_nbi(shmem_ctx_t ctx, uint64_t *target, const uint64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_size_get_nbi(shmem_ctx_t ctx, size_t *target, const size_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_ptrdiff_get_nbi(shmem_ctx_t ctx, ptrdiff_t *target, const ptrdiff_t *source, size_t len, int pe);

OSHMEM_DECLSPEC  void shmem_char_get_nbi(char *target, const char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_short_get_nbi(short *target, const short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int_get_nbi(int *target, const int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_long_get_nbi(long *target, const long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_longlong_get_nbi(long long *target, const long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_schar_get_nbi(signed char *target, const signed char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uchar_get_nbi(unsigned char *target, const unsigned char *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ushort_get_nbi(unsigned short *target, const unsigned short *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint_get_nbi(unsigned int *target, const unsigned int *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ulong_get_nbi(unsigned long *target, const unsigned long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ulonglong_get_nbi(unsigned long long *target, const unsigned long long *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_float_get_nbi(float *target, const float *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_double_get_nbi(double *target, const double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_longdouble_get_nbi(long double *target, const long double *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int8_get_nbi(int8_t *target, const int8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int16_get_nbi(int16_t *target, const int16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int32_get_nbi(int32_t *target, const int32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_int64_get_nbi(int64_t *target, const int64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint8_get_nbi(uint8_t *target, const uint8_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint16_get_nbi(uint16_t *target, const uint16_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint32_get_nbi(uint32_t *target, const uint32_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_uint64_get_nbi(uint64_t *target, const uint64_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_size_get_nbi(size_t *target, const size_t *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ptrdiff_get_nbi(ptrdiff_t *target, const ptrdiff_t *source, size_t len, int pe);
#if OSHMEM_HAVE_C11
#define shmem_get_nbi(...)                                           \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                     \
            shmem_ctx_t: _Generic(&*(__OSHMEM_VAR_ARG2(__VA_ARGS__)), \
                char*:        shmem_ctx_char_get_nbi,                \
                short*:       shmem_ctx_short_get_nbi,               \
                int*:         shmem_ctx_int_get_nbi,                 \
                long*:        shmem_ctx_long_get_nbi,                \
                long long*:   shmem_ctx_longlong_get_nbi,            \
                signed char*:        shmem_ctx_schar_get_nbi,        \
                unsigned char*:      shmem_ctx_uchar_get_nbi,        \
                unsigned short*:     shmem_ctx_ushort_get_nbi,       \
                unsigned int*:       shmem_ctx_uint_get_nbi,         \
                unsigned long*:      shmem_ctx_ulong_get_nbi,        \
                unsigned long long*: shmem_ctx_ulonglong_get_nbi,    \
                float*:       shmem_ctx_float_get_nbi,               \
                double*:      shmem_ctx_double_get_nbi,              \
                long double*: shmem_ctx_longdouble_get_nbi,          \
                default:      __oshmem_datatype_ignore),             \
            char*:        shmem_char_get_nbi,                        \
            short*:       shmem_short_get_nbi,                       \
            int*:         shmem_int_get_nbi,                         \
            long*:        shmem_long_get_nbi,                        \
            long long*:   shmem_longlong_get_nbi,                    \
            signed char*:        shmem_schar_get_nbi,                \
            unsigned char*:      shmem_uchar_get_nbi,                \
            unsigned short*:     shmem_ushort_get_nbi,               \
            unsigned int*:       shmem_uint_get_nbi,                 \
            unsigned long*:      shmem_ulong_get_nbi,                \
            unsigned long long*: shmem_ulonglong_get_nbi,            \
            float*:       shmem_float_get_nbi,                       \
            double*:      shmem_double_get_nbi,                      \
            long double*: shmem_longdouble_get_nbi)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC  void shmem_ctx_get8_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_get16_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_get32_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_get64_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_get128_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_ctx_getmem_nbi(shmem_ctx_t ctx, void *target, const void *source, size_t len, int pe);

OSHMEM_DECLSPEC  void shmem_get8_nbi(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_get16_nbi(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_get32_nbi(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_get64_nbi(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_get128_nbi(void *target, const void *source, size_t len, int pe);
OSHMEM_DECLSPEC  void shmem_getmem_nbi(void *target, const void *source, size_t len, int pe);

/*
 * Atomic operations
 */
/* Atomic swap */
OSHMEM_DECLSPEC int shmem_ctx_int_atomic_swap(shmem_ctx_t ctx, int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_ctx_long_atomic_swap(shmem_ctx_t ctx, long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_ctx_longlong_atomic_swap(shmem_ctx_t ctx, long long *target, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_ctx_uint_atomic_swap(shmem_ctx_t ctx, unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ctx_ulong_atomic_swap(shmem_ctx_t ctx, unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ctx_ulonglong_atomic_swap(shmem_ctx_t ctx, unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC float shmem_ctx_float_atomic_swap(shmem_ctx_t ctx, float *target, float value, int pe);
OSHMEM_DECLSPEC double shmem_ctx_double_atomic_swap(shmem_ctx_t ctx, double *target, double value, int pe);

OSHMEM_DECLSPEC int shmem_int_atomic_swap(int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_long_atomic_swap(long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_atomic_swap(long long*target, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_uint_atomic_swap(unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ulong_atomic_swap(unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ulonglong_atomic_swap(unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC float shmem_float_atomic_swap(float *target, float value, int pe);
OSHMEM_DECLSPEC double shmem_double_atomic_swap(double *target, double value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_atomic_swap(...)                                       \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                     \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)), \
                int*:         shmem_ctx_int_atomic_swap,             \
                long*:        shmem_ctx_long_atomic_swap,            \
                long long*:   shmem_ctx_longlong_atomic_swap,        \
                unsigned int*:       shmem_ctx_uint_atomic_swap,     \
                unsigned long*:      shmem_ctx_ulong_atomic_swap,    \
                unsigned long long*: shmem_ctx_ulonglong_atomic_swap,\
                float*:       shmem_ctx_float_atomic_swap,           \
                double*:      shmem_ctx_double_atomic_swap,          \
                default:      __oshmem_datatype_ignore),             \
            int*:         shmem_int_atomic_swap,                     \
            long*:        shmem_long_atomic_swap,                    \
            long long*:   shmem_longlong_atomic_swap,                \
            unsigned int*:       shmem_uint_atomic_swap,             \
            unsigned long*:      shmem_ulong_atomic_swap,            \
            unsigned long long*: shmem_ulonglong_atomic_swap,        \
            float*:       shmem_float_atomic_swap,                   \
            double*:      shmem_double_atomic_swap)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC int shmem_int_swap(int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_long_swap(long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_swap(long long*target, long long value, int pe);
OSHMEM_DECLSPEC float shmem_float_swap(float *target, float value, int pe);
OSHMEM_DECLSPEC double shmem_double_swap(double *target, double value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_swap(dst, val, pe)                             \
    _Generic(&*(dst),                                        \
            int*:         shmem_int_swap,                    \
            long*:        shmem_long_swap,                   \
            long long*:   shmem_longlong_swap,               \
            float*:       shmem_float_swap,                  \
            double*:      shmem_double_swap)(dst, val, pe)
#endif

/* Atomic set */
OSHMEM_DECLSPEC void shmem_ctx_int_atomic_set(shmem_ctx_t ctx, int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_long_atomic_set(shmem_ctx_t ctx, long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_longlong_atomic_set(shmem_ctx_t ctx, long long *target, long long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint_atomic_set(shmem_ctx_t ctx, unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulong_atomic_set(shmem_ctx_t ctx, unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulonglong_atomic_set(shmem_ctx_t ctx, unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_float_atomic_set(shmem_ctx_t ctx, float *target, float value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_double_atomic_set(shmem_ctx_t ctx, double *target, double value, int pe);

OSHMEM_DECLSPEC void shmem_int_atomic_set(int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_long_atomic_set(long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_longlong_atomic_set(long long *target, long long value, int pe);
OSHMEM_DECLSPEC void shmem_uint_atomic_set(unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC void shmem_ulong_atomic_set(unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC void shmem_ulonglong_atomic_set(unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC void shmem_float_atomic_set(float *target, float value, int pe);
OSHMEM_DECLSPEC void shmem_double_atomic_set(double *target, double value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_atomic_set(...)                                       \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                    \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)),\
                int*:         shmem_ctx_int_atomic_set,             \
                long*:        shmem_ctx_long_atomic_set,            \
                long long*:   shmem_ctx_longlong_atomic_set,        \
                unsigned int*:       shmem_ctx_uint_atomic_set,     \
                unsigned long*:      shmem_ctx_ulong_atomic_set,    \
                unsigned long long*: shmem_ctx_ulonglong_atomic_set,\
                float*:       shmem_ctx_float_atomic_set,           \
                double*:      shmem_ctx_double_atomic_set,          \
                default:      __oshmem_datatype_ignore),            \
            int*:         shmem_int_atomic_set,                     \
            long*:        shmem_long_atomic_set,                    \
            long long*:   shmem_longlong_atomic_set,                \
            unsigned int*:         shmem_uint_atomic_set,           \
            unsigned long*:        shmem_ulong_atomic_set,          \
            unsigned long long*:   shmem_ulonglong_atomic_set,      \
            float*:       shmem_float_atomic_set,                   \
            double*:      shmem_double_atomic_set)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC void shmem_int_set(int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_long_set(long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_longlong_set(long long *target, long long value, int pe);
OSHMEM_DECLSPEC void shmem_float_set(float *target, float value, int pe);
OSHMEM_DECLSPEC void shmem_double_set(double *target, double value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_set(dst, val, pe)                             \
    _Generic(&*(dst),                                       \
            int*:         shmem_int_set,                    \
            long*:        shmem_long_set,                   \
            long long*:   shmem_longlong_set,               \
            float*:       shmem_float_set,                  \
            double*:      shmem_double_set)(dst, val, pe)
#endif

/* Atomic conditional swap */
OSHMEM_DECLSPEC int shmem_ctx_int_atomic_compare_swap(shmem_ctx_t ctx, int *target, int cond, int value, int pe);
OSHMEM_DECLSPEC long shmem_ctx_long_atomic_compare_swap(shmem_ctx_t ctx, long *target, long cond, long value, int pe);
OSHMEM_DECLSPEC long long shmem_ctx_longlong_atomic_compare_swap(shmem_ctx_t ctx, long long *target, long long cond, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_ctx_uint_atomic_compare_swap(shmem_ctx_t ctx, unsigned int *target, unsigned int cond, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ctx_ulong_atomic_compare_swap(shmem_ctx_t ctx, unsigned long *target, unsigned long cond, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ctx_ulonglong_atomic_compare_swap(shmem_ctx_t ctx, unsigned long long *target, unsigned long long cond, unsigned long long value, int pe);

OSHMEM_DECLSPEC int shmem_int_atomic_compare_swap(int *target, int cond, int value, int pe);
OSHMEM_DECLSPEC long shmem_long_atomic_compare_swap(long *target, long cond, long value, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_atomic_compare_swap(long long *target, long long cond, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_uint_atomic_compare_swap(unsigned int *target, unsigned int cond, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ulong_atomic_compare_swap(unsigned long *target, unsigned long cond, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ulonglong_atomic_compare_swap(unsigned long long *target, unsigned long long cond, unsigned long long value, int pe);

#if OSHMEM_HAVE_C11
#define shmem_atomic_compare_swap(...)                                \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                      \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)),  \
                int*:         shmem_ctx_int_atomic_compare_swap,      \
                long*:        shmem_ctx_long_atomic_compare_swap,     \
                long long*:   shmem_ctx_longlong_atomic_compare_swap, \
                unsigned int*:       shmem_ctx_uint_atomic_compare_swap,      \
                unsigned long*:      shmem_ctx_ulong_atomic_compare_swap,     \
                unsigned long long*: shmem_ctx_ulonglong_atomic_compare_swap, \
                default:      __oshmem_datatype_ignore),              \
            int*:         shmem_int_atomic_compare_swap,              \
            long*:        shmem_long_atomic_compare_swap,             \
            long long*:   shmem_longlong_atomic_compare_swap,         \
            unsigned int*:       shmem_uint_atomic_compare_swap,      \
            unsigned long*:      shmem_ulong_atomic_compare_swap,     \
            unsigned long long*: shmem_ulonglong_atomic_compare_swap)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC int shmem_int_cswap(int *target, int cond, int value, int pe);
OSHMEM_DECLSPEC long shmem_long_cswap(long *target, long cond, long value, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_cswap(long long *target, long long cond, long long value, int pe);

#if OSHMEM_HAVE_C11
#define shmem_cswap(dst, cond, val, pe)                       \
    _Generic(&*(dst),                                         \
            int*:         shmem_int_cswap,                    \
            long*:        shmem_long_cswap,                   \
            long long*:   shmem_longlong_cswap)(dst, cond, val, pe)
#endif

/* Atomic Fetch&Add */
OSHMEM_DECLSPEC int shmem_ctx_int_atomic_fetch_add(shmem_ctx_t ctx, int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_ctx_long_atomic_fetch_add(shmem_ctx_t ctx, long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_ctx_longlong_atomic_fetch_add(shmem_ctx_t ctx, long long *target, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_ctx_uint_atomic_fetch_add(shmem_ctx_t ctx, unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ctx_ulong_atomic_fetch_add(shmem_ctx_t ctx, unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ctx_ulonglong_atomic_fetch_add(shmem_ctx_t ctx, unsigned long long *target, unsigned long long value, int pe);

OSHMEM_DECLSPEC int shmem_int_atomic_fetch_add(int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_long_atomic_fetch_add(long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_atomic_fetch_add(long long *target, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_uint_atomic_fetch_add(unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ulong_atomic_fetch_add(unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ulonglong_atomic_fetch_add(unsigned long long *target, unsigned long long value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_atomic_fetch_add(...)                                        \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                           \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)),       \
                int*:         shmem_ctx_int_atomic_fetch_add,              \
                long*:        shmem_ctx_long_atomic_fetch_add,             \
                long long*:   shmem_ctx_longlong_atomic_fetch_add,         \
                unsigned int*:       shmem_ctx_uint_atomic_fetch_add,      \
                unsigned long*:      shmem_ctx_ulong_atomic_fetch_add,     \
                unsigned long long*: shmem_ctx_ulonglong_atomic_fetch_add, \
                default:      __oshmem_datatype_ignore),                   \
            int*:         shmem_int_atomic_fetch_add,                      \
            long*:        shmem_long_atomic_fetch_add,                     \
            long long*:   shmem_longlong_atomic_fetch_add,                 \
            unsigned int*:       shmem_uint_atomic_fetch_add,              \
            unsigned long*:      shmem_ulong_atomic_fetch_add,             \
            unsigned long long*: shmem_ulonglong_atomic_fetch_add)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC int shmem_int_fadd(int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_long_fadd(long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_fadd(long long *target, long long value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_fadd(dst, val, pe)                             \
    _Generic(&*(dst),                                        \
            int*:         shmem_int_fadd,                    \
            long*:        shmem_long_fadd,                   \
            long long*:   shmem_longlong_fadd)(dst, val, pe)
#endif

/* Atomic Fetch&And */
OSHMEM_DECLSPEC int shmem_ctx_int_atomic_fetch_and(shmem_ctx_t ctx, int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_ctx_long_atomic_fetch_and(shmem_ctx_t ctx, long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_ctx_longlong_atomic_fetch_and(shmem_ctx_t ctx, long long *target, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_ctx_uint_atomic_fetch_and(shmem_ctx_t ctx, unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ctx_ulong_atomic_fetch_and(shmem_ctx_t ctx, unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ctx_ulonglong_atomic_fetch_and(shmem_ctx_t ctx, unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC int32_t shmem_ctx_int32_atomic_fetch_and(shmem_ctx_t ctx, int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmem_ctx_int64_atomic_fetch_and(shmem_ctx_t ctx, int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t shmem_ctx_uint32_atomic_fetch_and(shmem_ctx_t ctx, uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t shmem_ctx_uint64_atomic_fetch_and(shmem_ctx_t ctx, uint64_t *target, uint64_t value, int pe);

OSHMEM_DECLSPEC int shmem_int_atomic_fetch_and(int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_long_atomic_fetch_and(long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_atomic_fetch_and(long long *target, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_uint_atomic_fetch_and(unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ulong_atomic_fetch_and(unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ulonglong_atomic_fetch_and(unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC int32_t shmem_int32_atomic_fetch_and(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmem_int64_atomic_fetch_and(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t shmem_uint32_atomic_fetch_and(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t shmem_uint64_atomic_fetch_and(uint64_t *target, uint64_t value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_atomic_fetch_and(...)                                           \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                              \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)),          \
                int*:         shmem_ctx_int_atomic_fetch_and,                 \
                long*:        shmem_ctx_long_atomic_fetch_and,                \
                long long*:   shmem_ctx_longlong_atomic_fetch_and,            \
                unsigned int*:         shmem_ctx_uint_atomic_fetch_and,       \
                unsigned long*:        shmem_ctx_ulong_atomic_fetch_and,      \
                unsigned long long*:   shmem_ctx_ulonglong_atomic_fetch_and,  \
                default:               __oshmem_datatype_ignore),             \
            int*:         shmem_int_atomic_fetch_and,                         \
            long*:        shmem_long_atomic_fetch_and,                        \
            long long*:   shmem_longlong_atomic_fetch_and,                    \
            unsigned int*:         shmem_uint_atomic_fetch_and,               \
            unsigned long*:        shmem_ulong_atomic_fetch_and,              \
            unsigned long long*:   shmem_ulonglong_atomic_fetch_and)(__VA_ARGS__)
#endif

/* Atomic Fetch&Or */
OSHMEM_DECLSPEC int shmem_ctx_int_atomic_fetch_or(shmem_ctx_t ctx, int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_ctx_long_atomic_fetch_or(shmem_ctx_t ctx, long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_ctx_longlong_atomic_fetch_or(shmem_ctx_t ctx, long long *target, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_ctx_uint_atomic_fetch_or(shmem_ctx_t ctx, unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ctx_ulong_atomic_fetch_or(shmem_ctx_t ctx, unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ctx_ulonglong_atomic_fetch_or(shmem_ctx_t ctx, unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC int32_t shmem_ctx_int32_atomic_fetch_or(shmem_ctx_t ctx, int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmem_ctx_int64_atomic_fetch_or(shmem_ctx_t ctx, int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t shmem_ctx_uint32_atomic_fetch_or(shmem_ctx_t ctx, uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t shmem_ctx_uint64_atomic_fetch_or(shmem_ctx_t ctx, uint64_t *target, uint64_t value, int pe);

OSHMEM_DECLSPEC int shmem_int_atomic_fetch_or(int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_long_atomic_fetch_or(long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_atomic_fetch_or(long long *target, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_uint_atomic_fetch_or(unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ulong_atomic_fetch_or(unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ulonglong_atomic_fetch_or(unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC int32_t shmem_int32_atomic_fetch_or(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmem_int64_atomic_fetch_or(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t shmem_uint32_atomic_fetch_or(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t shmem_uint64_atomic_fetch_or(uint64_t *target, uint64_t value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_atomic_fetch_or(...)                                           \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                             \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)),         \
                int*:         shmem_ctx_int_atomic_fetch_or,                 \
                long*:        shmem_ctx_long_atomic_fetch_or,                \
                long long*:   shmem_ctx_longlong_atomic_fetch_or,            \
                unsigned int*:         shmem_ctx_uint_atomic_fetch_or,       \
                unsigned long*:        shmem_ctx_ulong_atomic_fetch_or,      \
                unsigned long long*:   shmem_ctx_ulonglong_atomic_fetch_or,  \
                default:               __oshmem_datatype_ignore),            \
            int*:         shmem_int_atomic_fetch_or,                         \
            long*:        shmem_long_atomic_fetch_or,                        \
            long long*:   shmem_longlong_atomic_fetch_or,                    \
            unsigned int*:         shmem_uint_atomic_fetch_or,               \
            unsigned long*:        shmem_ulong_atomic_fetch_or,              \
            unsigned long long*:   shmem_ulonglong_atomic_fetch_or)(__VA_ARGS__)
#endif

/* Atomic Fetch&Xor */
OSHMEM_DECLSPEC int shmem_ctx_int_atomic_fetch_xor(shmem_ctx_t ctx, int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_ctx_long_atomic_fetch_xor(shmem_ctx_t ctx, long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_ctx_longlong_atomic_fetch_xor(shmem_ctx_t ctx, long long *target, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_ctx_uint_atomic_fetch_xor(shmem_ctx_t ctx, unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ctx_ulong_atomic_fetch_xor(shmem_ctx_t ctx, unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ctx_ulonglong_atomic_fetch_xor(shmem_ctx_t ctx, unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC int32_t shmem_ctx_int32_atomic_fetch_xor(shmem_ctx_t ctx, int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmem_ctx_int64_atomic_fetch_xor(shmem_ctx_t ctx, int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t shmem_ctx_uint32_atomic_fetch_xor(shmem_ctx_t ctx, uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t shmem_ctx_uint64_atomic_fetch_xor(shmem_ctx_t ctx, uint64_t *target, uint64_t value, int pe);

OSHMEM_DECLSPEC int shmem_int_atomic_fetch_xor(int *target, int value, int pe);
OSHMEM_DECLSPEC long shmem_long_atomic_fetch_xor(long *target, long value, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_atomic_fetch_xor(long long *target, long long value, int pe);
OSHMEM_DECLSPEC unsigned int shmem_uint_atomic_fetch_xor(unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ulong_atomic_fetch_xor(unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ulonglong_atomic_fetch_xor(unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC int32_t shmem_int32_atomic_fetch_xor(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC int64_t shmem_int64_atomic_fetch_xor(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC uint32_t shmem_uint32_atomic_fetch_xor(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC uint64_t shmem_uint64_atomic_fetch_xor(uint64_t *target, uint64_t value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_atomic_fetch_xor(...)                                           \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                              \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)),          \
                int*:         shmem_ctx_int_atomic_fetch_xor,                 \
                long*:        shmem_ctx_long_atomic_fetch_xor,                \
                long long*:   shmem_ctx_longlong_atomic_fetch_xor,            \
                unsigned int*:         shmem_ctx_uint_atomic_fetch_xor,       \
                unsigned long*:        shmem_ctx_ulong_atomic_fetch_xor,      \
                unsigned long long*:   shmem_ctx_ulonglong_atomic_fetch_xor,  \
                default:               __oshmem_datatype_ignore),             \
            int*:         shmem_int_atomic_fetch_xor,                         \
            long*:        shmem_long_atomic_fetch_xor,                        \
            long long*:   shmem_longlong_atomic_fetch_xor,                    \
            unsigned int*:         shmem_uint_atomic_fetch_xor,               \
            unsigned long*:        shmem_ulong_atomic_fetch_xor,              \
            unsigned long long*:   shmem_ulonglong_atomic_fetch_xor)(__VA_ARGS__)
#endif

/* Atomic Fetch */
OSHMEM_DECLSPEC int shmem_ctx_int_atomic_fetch(shmem_ctx_t ctx, const int *target, int pe);
OSHMEM_DECLSPEC long shmem_ctx_long_atomic_fetch(shmem_ctx_t ctx, const long *target, int pe);
OSHMEM_DECLSPEC long long shmem_ctx_longlong_atomic_fetch(shmem_ctx_t ctx, const long long *target, int pe);
OSHMEM_DECLSPEC unsigned int shmem_ctx_uint_atomic_fetch(shmem_ctx_t ctx, const unsigned int *target, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ctx_ulong_atomic_fetch(shmem_ctx_t ctx, const unsigned long *target, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ctx_ulonglong_atomic_fetch(shmem_ctx_t ctx, const unsigned long long *target, int pe);
OSHMEM_DECLSPEC float shmem_ctx_float_atomic_fetch(shmem_ctx_t ctx, const float *target, int pe);
OSHMEM_DECLSPEC double shmem_ctx_double_atomic_fetch(shmem_ctx_t ctx, const double *target, int pe);

OSHMEM_DECLSPEC int shmem_int_atomic_fetch(const int *target, int pe);
OSHMEM_DECLSPEC long shmem_long_atomic_fetch(const long *target, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_atomic_fetch(const long long *target, int pe);
OSHMEM_DECLSPEC unsigned int shmem_uint_atomic_fetch(const unsigned int *target, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ulong_atomic_fetch(const unsigned long *target, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ulonglong_atomic_fetch(const unsigned long long *target, int pe);
OSHMEM_DECLSPEC float shmem_float_atomic_fetch(const float *target, int pe);
OSHMEM_DECLSPEC double shmem_double_atomic_fetch(const double *target, int pe);
#if OSHMEM_HAVE_C11
#define shmem_atomic_fetch(...)                                      \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                     \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)), \
                int*:         shmem_ctx_int_atomic_fetch,            \
                long*:        shmem_ctx_long_atomic_fetch,           \
                long long*:   shmem_ctx_longlong_atomic_fetch,       \
                unsigned int*:       shmem_ctx_uint_atomic_fetch,      \
                unsigned long*:      shmem_ctx_ulong_atomic_fetch,     \
                unsigned long long*: shmem_ctx_ulonglong_atomic_fetch, \
                float*:       shmem_ctx_float_atomic_fetch,          \
                double*:      shmem_ctx_double_atomic_fetch,         \
                default:      __oshmem_datatype_ignore),             \
            int*:         shmem_int_atomic_fetch,                    \
            long*:        shmem_long_atomic_fetch,                   \
            long long*:   shmem_longlong_atomic_fetch,               \
            unsigned int*:       shmem_uint_atomic_fetch,            \
            unsigned long*:      shmem_ulong_atomic_fetch,           \
            unsigned long long*: shmem_ulonglong_atomic_fetch,       \
            float*:       shmem_float_atomic_fetch,                  \
            double*:      shmem_double_atomic_fetch)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC int shmem_int_fetch(const int *target, int pe);
OSHMEM_DECLSPEC long shmem_long_fetch(const long *target, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_fetch(const long long *target, int pe);
OSHMEM_DECLSPEC float shmem_float_fetch(const float *target, int pe);
OSHMEM_DECLSPEC double shmem_double_fetch(const double *target, int pe);
#if OSHMEM_HAVE_C11
#define shmem_fetch(dst, pe)                             \
    _Generic(&*(dst),                                    \
            int*:         shmem_int_fetch,               \
            long*:        shmem_long_fetch,              \
            long long*:   shmem_longlong_fetch,          \
            float*:       shmem_float_fetch,             \
            double*:      shmem_double_fetch)(dst, pe)
#endif

/* Atomic Fetch&Inc */
OSHMEM_DECLSPEC int shmem_ctx_int_atomic_fetch_inc(shmem_ctx_t ctx, int *target, int pe);
OSHMEM_DECLSPEC long shmem_ctx_long_atomic_fetch_inc(shmem_ctx_t ctx, long *target, int pe);
OSHMEM_DECLSPEC long long shmem_ctx_longlong_atomic_fetch_inc(shmem_ctx_t ctx, long long *target, int pe);
OSHMEM_DECLSPEC unsigned int shmem_ctx_uint_atomic_fetch_inc(shmem_ctx_t ctx, unsigned int *target, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ctx_ulong_atomic_fetch_inc(shmem_ctx_t ctx, unsigned long *target, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ctx_ulonglong_atomic_fetch_inc(shmem_ctx_t ctx, unsigned long long *target, int pe);

OSHMEM_DECLSPEC int shmem_int_atomic_fetch_inc(int *target, int pe);
OSHMEM_DECLSPEC long shmem_long_atomic_fetch_inc(long *target, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_atomic_fetch_inc(long long *target, int pe);
OSHMEM_DECLSPEC unsigned int shmem_uint_atomic_fetch_inc(unsigned int *target, int pe);
OSHMEM_DECLSPEC unsigned long shmem_ulong_atomic_fetch_inc(unsigned long *target, int pe);
OSHMEM_DECLSPEC unsigned long long shmem_ulonglong_atomic_fetch_inc(unsigned long long *target, int pe);
#if OSHMEM_HAVE_C11
#define shmem_atomic_fetch_inc(...)                                 \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                     \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)), \
                int*:         shmem_ctx_int_atomic_fetch_inc,       \
                long*:        shmem_ctx_long_atomic_fetch_inc,      \
                long long*:   shmem_ctx_longlong_atomic_fetch_inc,  \
                unsigned int*:       shmem_ctx_uint_atomic_fetch_inc,      \
                unsigned long*:      shmem_ctx_ulong_atomic_fetch_inc,     \
                unsigned long long*: shmem_ctx_ulonglong_atomic_fetch_inc, \
                default:      __oshmem_datatype_ignore),             \
            int*:         shmem_int_atomic_fetch_inc,               \
            long*:        shmem_long_atomic_fetch_inc,              \
            long long*:   shmem_longlong_atomic_fetch_inc,          \
            unsigned int*:       shmem_uint_atomic_fetch_inc,       \
            unsigned long*:      shmem_ulong_atomic_fetch_inc,      \
            unsigned long long*: shmem_ulonglong_atomic_fetch_inc)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC int shmem_int_finc(int *target, int pe);
OSHMEM_DECLSPEC long shmem_long_finc(long *target, int pe);
OSHMEM_DECLSPEC long long shmem_longlong_finc(long long *target, int pe);
#if OSHMEM_HAVE_C11
#define shmem_finc(dst, pe)                            \
    _Generic(&*(dst),                                  \
            int*:         shmem_int_finc,              \
            long*:        shmem_long_finc,             \
            long long*:   shmem_longlong_finc)(dst, pe)
#endif

/* Atomic Add */
OSHMEM_DECLSPEC void shmem_ctx_int_atomic_add(shmem_ctx_t ctx, int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_long_atomic_add(shmem_ctx_t ctx, long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_longlong_atomic_add(shmem_ctx_t ctx, long long *target, long long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint_atomic_add(shmem_ctx_t ctx, unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulong_atomic_add(shmem_ctx_t ctx, unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulonglong_atomic_add(shmem_ctx_t ctx, unsigned long long *target, unsigned long long value, int pe);

OSHMEM_DECLSPEC void shmem_int_atomic_add(int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_long_atomic_add(long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_longlong_atomic_add(long long *target, long long value, int pe);
OSHMEM_DECLSPEC void shmem_uint_atomic_add(unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC void shmem_ulong_atomic_add(unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC void shmem_ulonglong_atomic_add(unsigned long long *target, unsigned long long value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_atomic_add(...)                                        \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                     \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)), \
                int*:         shmem_ctx_int_atomic_add,              \
                long*:        shmem_ctx_long_atomic_add,             \
                long long*:   shmem_ctx_longlong_atomic_add,         \
                unsigned int*:       shmem_ctx_uint_atomic_add,      \
                unsigned long*:      shmem_ctx_ulong_atomic_add,     \
                unsigned long long*: shmem_ctx_ulonglong_atomic_add, \
                default:      __oshmem_datatype_ignore),             \
            int*:         shmem_int_atomic_add,                      \
            long*:        shmem_long_atomic_add,                     \
            long long*:   shmem_longlong_atomic_add,                 \
            unsigned int*:       shmem_uint_atomic_add,              \
            unsigned long*:      shmem_ulong_atomic_add,             \
            unsigned long long*: shmem_ulonglong_atomic_add)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC void shmem_int_add(int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_long_add(long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_longlong_add(long long *target, long long value, int pe);
#if OSHMEM_HAVE_C11
#define shmem_add(dst, val, pe)                             \
    _Generic(&*(dst),                                       \
            int*:         shmem_int_add,                    \
            long*:        shmem_long_add,                   \
            long long*:   shmem_longlong_add)(dst, val, pe)
#endif

/* Atomic And */
OSHMEM_DECLSPEC void shmem_ctx_int_atomic_and(shmem_ctx_t ctx, int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_long_atomic_and(shmem_ctx_t ctx, long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_longlong_atomic_and(shmem_ctx_t ctx, long long *target, long long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint_atomic_and(shmem_ctx_t ctx, unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulong_atomic_and(shmem_ctx_t ctx, unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulonglong_atomic_and(shmem_ctx_t ctx, unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int32_atomic_and(shmem_ctx_t ctx, int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int64_atomic_and(shmem_ctx_t ctx, int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint32_atomic_and(shmem_ctx_t ctx, uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint64_atomic_and(shmem_ctx_t ctx, uint64_t *target, uint64_t value, int pe);

OSHMEM_DECLSPEC void shmem_int_atomic_and(int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_long_atomic_and(long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_longlong_atomic_and(long long *target, long long value, int pe);
OSHMEM_DECLSPEC void shmem_uint_atomic_and(unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC void shmem_ulong_atomic_and(unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC void shmem_ulonglong_atomic_and(unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC void shmem_int32_atomic_and(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void shmem_int64_atomic_and(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void shmem_uint32_atomic_and(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void shmem_uint64_atomic_and(uint64_t *target, uint64_t value, int pe);

#if OSHMEM_HAVE_C11
#define shmem_atomic_and(...)                                          \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                       \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)),   \
                int*:         shmem_ctx_int_atomic_and,                \
                long*:        shmem_ctx_long_atomic_and,               \
                long long*:   shmem_ctx_longlong_atomic_and,           \
                unsigned int*:         shmem_ctx_uint_atomic_and,      \
                unsigned long*:        shmem_ctx_ulong_atomic_and,     \
                unsigned long long*:   shmem_ctx_ulonglong_atomic_and, \
                default:               __oshmem_datatype_ignore),      \
            int*:         shmem_int_atomic_and,                        \
            long*:        shmem_long_atomic_and,                       \
            long long*:   shmem_longlong_atomic_and,                   \
            unsigned int*:         shmem_uint_atomic_and,              \
            unsigned long*:        shmem_ulong_atomic_and,             \
            unsigned long long*:   shmem_ulonglong_atomic_and)(__VA_ARGS__)
#endif

/* Atomic Or */
OSHMEM_DECLSPEC void shmem_ctx_int_atomic_or(shmem_ctx_t ctx, int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_long_atomic_or(shmem_ctx_t ctx, long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_longlong_atomic_or(shmem_ctx_t ctx, long long *target, long long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint_atomic_or(shmem_ctx_t ctx, unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulong_atomic_or(shmem_ctx_t ctx, unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulonglong_atomic_or(shmem_ctx_t ctx, unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int32_atomic_or(shmem_ctx_t ctx, int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int64_atomic_or(shmem_ctx_t ctx, int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint32_atomic_or(shmem_ctx_t ctx, uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint64_atomic_or(shmem_ctx_t ctx, uint64_t *target, uint64_t value, int pe);

OSHMEM_DECLSPEC void shmem_int_atomic_or(int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_long_atomic_or(long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_longlong_atomic_or(long long *target, long long value, int pe);
OSHMEM_DECLSPEC void shmem_uint_atomic_or(unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC void shmem_ulong_atomic_or(unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC void shmem_ulonglong_atomic_or(unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC void shmem_int32_atomic_or(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void shmem_int64_atomic_or(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void shmem_uint32_atomic_or(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void shmem_uint64_atomic_or(uint64_t *target, uint64_t value, int pe);

#if OSHMEM_HAVE_C11
#define shmem_atomic_or(...)                                          \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                      \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)),  \
                int*:         shmem_ctx_int_atomic_or,                \
                long*:        shmem_ctx_long_atomic_or,               \
                long long*:   shmem_ctx_longlong_atomic_or,           \
                unsigned int*:         shmem_ctx_uint_atomic_or,      \
                unsigned long*:        shmem_ctx_ulong_atomic_or,     \
                unsigned long long*:   shmem_ctx_ulonglong_atomic_or, \
                default:               __oshmem_datatype_ignore),     \
            int*:         shmem_int_atomic_or,                        \
            long*:        shmem_long_atomic_or,                       \
            long long*:   shmem_longlong_atomic_or,                   \
            unsigned int*:         shmem_uint_atomic_or,              \
            unsigned long*:        shmem_ulong_atomic_or,             \
            unsigned long long*:   shmem_ulonglong_atomic_or)(__VA_ARGS__)
#endif

/* Atomic Xor */
OSHMEM_DECLSPEC void shmem_ctx_int_atomic_xor(shmem_ctx_t ctx, int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_long_atomic_xor(shmem_ctx_t ctx, long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_longlong_atomic_xor(shmem_ctx_t ctx, long long *target, long long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint_atomic_xor(shmem_ctx_t ctx, unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulong_atomic_xor(shmem_ctx_t ctx, unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulonglong_atomic_xor(shmem_ctx_t ctx, unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int32_atomic_xor(shmem_ctx_t ctx, int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_int64_atomic_xor(shmem_ctx_t ctx, int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint32_atomic_xor(shmem_ctx_t ctx, uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint64_atomic_xor(shmem_ctx_t ctx, uint64_t *target, uint64_t value, int pe);

OSHMEM_DECLSPEC void shmem_int_atomic_xor(int *target, int value, int pe);
OSHMEM_DECLSPEC void shmem_long_atomic_xor(long *target, long value, int pe);
OSHMEM_DECLSPEC void shmem_longlong_atomic_xor(long long *target, long long value, int pe);
OSHMEM_DECLSPEC void shmem_uint_atomic_xor(unsigned int *target, unsigned int value, int pe);
OSHMEM_DECLSPEC void shmem_ulong_atomic_xor(unsigned long *target, unsigned long value, int pe);
OSHMEM_DECLSPEC void shmem_ulonglong_atomic_xor(unsigned long long *target, unsigned long long value, int pe);
OSHMEM_DECLSPEC void shmem_int32_atomic_xor(int32_t *target, int32_t value, int pe);
OSHMEM_DECLSPEC void shmem_int64_atomic_xor(int64_t *target, int64_t value, int pe);
OSHMEM_DECLSPEC void shmem_uint32_atomic_xor(uint32_t *target, uint32_t value, int pe);
OSHMEM_DECLSPEC void shmem_uint64_atomic_xor(uint64_t *target, uint64_t value, int pe);

#if OSHMEM_HAVE_C11
#define shmem_atomic_xor(...)                                          \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                       \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)),   \
                int*:         shmem_ctx_int_atomic_xor,                \
                long*:        shmem_ctx_long_atomic_xor,               \
                long long*:   shmem_ctx_longlong_atomic_xor,           \
                unsigned int*:         shmem_ctx_uint_atomic_xor,      \
                unsigned long*:        shmem_ctx_ulong_atomic_xor,     \
                unsigned long long*:   shmem_ctx_ulonglong_atomic_xor, \
                default:               __oshmem_datatype_ignore),      \
            int*:         shmem_int_atomic_xor,                        \
            long*:        shmem_long_atomic_xor,                       \
            long long*:   shmem_longlong_atomic_xor,                   \
            unsigned int*:         shmem_uint_atomic_xor,              \
            unsigned long*:        shmem_ulong_atomic_xor,             \
            unsigned long long*:   shmem_ulonglong_atomic_xor)(__VA_ARGS__)
#endif

/* Atomic Inc */
OSHMEM_DECLSPEC void shmem_ctx_int_atomic_inc(shmem_ctx_t ctx, int *target, int pe);
OSHMEM_DECLSPEC void shmem_ctx_long_atomic_inc(shmem_ctx_t ctx, long *target, int pe);
OSHMEM_DECLSPEC void shmem_ctx_longlong_atomic_inc(shmem_ctx_t ctx, long long *target, int pe);
OSHMEM_DECLSPEC void shmem_ctx_uint_atomic_inc(shmem_ctx_t ctx, unsigned int *target, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulong_atomic_inc(shmem_ctx_t ctx, unsigned long *target, int pe);
OSHMEM_DECLSPEC void shmem_ctx_ulonglong_atomic_inc(shmem_ctx_t ctx, unsigned long long *target, int pe);

OSHMEM_DECLSPEC void shmem_int_atomic_inc(int *target, int pe);
OSHMEM_DECLSPEC void shmem_long_atomic_inc(long *target, int pe);
OSHMEM_DECLSPEC void shmem_longlong_atomic_inc(long long *target, int pe);
OSHMEM_DECLSPEC void shmem_uint_atomic_inc(unsigned int *target, int pe);
OSHMEM_DECLSPEC void shmem_ulong_atomic_inc(unsigned long *target, int pe);
OSHMEM_DECLSPEC void shmem_ulonglong_atomic_inc(unsigned long long *target, int pe);
#if OSHMEM_HAVE_C11
#define shmem_atomic_inc(...)                                       \
    _Generic(&*(__OSHMEM_VAR_ARG1(__VA_ARGS__)),                    \
            shmem_ctx_t:  _Generic((__OSHMEM_VAR_ARG2(__VA_ARGS__)),\
                int*:         shmem_ctx_int_atomic_inc,             \
                long*:        shmem_ctx_long_atomic_inc,            \
                long long*:   shmem_ctx_longlong_atomic_inc,        \
                unsigned int*:       shmem_ctx_uint_atomic_inc,     \
                unsigned long*:      shmem_ctx_ulong_atomic_inc,    \
                unsigned long long*: shmem_ctx_ulonglong_atomic_inc,\
                default:      __oshmem_datatype_ignore),            \
            int*:         shmem_int_atomic_inc,                     \
            long*:        shmem_long_atomic_inc,                    \
            long long*:   shmem_longlong_atomic_inc,                \
            unsigned int*:       shmem_uint_atomic_inc,             \
            unsigned long*:      shmem_ulong_atomic_inc,            \
            unsigned long long*: shmem_ulonglong_atomic_inc)(__VA_ARGS__)
#endif

OSHMEM_DECLSPEC void shmem_int_inc(int *target, int pe);
OSHMEM_DECLSPEC void shmem_long_inc(long *target, int pe);
OSHMEM_DECLSPEC void shmem_longlong_inc(long long *target, int pe);
#if OSHMEM_HAVE_C11
#define shmem_inc(dst, pe)                            \
    _Generic(&*(dst),                                 \
            int*:         shmem_int_inc,              \
            long*:        shmem_long_inc,             \
            long long*:   shmem_longlong_inc)(dst, pe)
#endif

/*
 * Lock functions
 */
OSHMEM_DECLSPEC void shmem_set_lock(volatile long *lock);
OSHMEM_DECLSPEC void shmem_clear_lock(volatile long *lock);
OSHMEM_DECLSPEC int shmem_test_lock(volatile long *lock);

/*
 * P2P sync routines
 */
OSHMEM_DECLSPEC  void shmem_short_wait(volatile short *addr, short value);
OSHMEM_DECLSPEC  void shmem_int_wait(volatile int *addr, int value);
OSHMEM_DECLSPEC  void shmem_long_wait(volatile long *addr, long value);
OSHMEM_DECLSPEC  void shmem_longlong_wait(volatile long long *addr, long long value);
OSHMEM_DECLSPEC  void shmem_wait(volatile long *addr, long value);

OSHMEM_DECLSPEC  void shmem_short_wait_until(volatile short *addr, int cmp, short value);
OSHMEM_DECLSPEC  void shmem_int_wait_until(volatile int *addr, int cmp, int value);
OSHMEM_DECLSPEC  void shmem_long_wait_until(volatile long *addr, int cmp, long value);
OSHMEM_DECLSPEC  void shmem_longlong_wait_until(volatile long long *addr, int cmp, long long value);
OSHMEM_DECLSPEC  void shmem_ushort_wait_until(volatile unsigned short *addr, int cmp, unsigned short value);
OSHMEM_DECLSPEC  void shmem_uint_wait_until(volatile unsigned int *addr, int cmp, unsigned int value);
OSHMEM_DECLSPEC  void shmem_ulong_wait_until(volatile unsigned long *addr, int cmp, unsigned long value);
OSHMEM_DECLSPEC  void shmem_ulonglong_wait_until(volatile unsigned long long *addr, int cmp, unsigned long long value);
OSHMEM_DECLSPEC  void shmem_int32_wait_until(volatile int32_t *addr, int cmp, int32_t value);
OSHMEM_DECLSPEC  void shmem_int64_wait_until(volatile int64_t *addr, int cmp, int64_t value);
OSHMEM_DECLSPEC  void shmem_uint32_wait_until(volatile uint32_t *addr, int cmp, uint32_t value);
OSHMEM_DECLSPEC  void shmem_uint64_wait_until(volatile uint64_t *addr, int cmp, uint64_t value);
OSHMEM_DECLSPEC  void shmem_size_wait_until(volatile size_t *addr, int cmp, size_t value);
OSHMEM_DECLSPEC  void shmem_ptrdiff_wait_until(volatile ptrdiff_t *addr, int cmp, ptrdiff_t value);
#if OSHMEM_HAVE_C11
#define shmem_wait_until(addr, cmp, value)                  \
    _Generic(&*(addr),                                      \
        short*:       shmem_short_wait_until,               \
        int*:         shmem_int_wait_until,                 \
        long*:        shmem_long_wait_until,                \
        long long*:   shmem_longlong_wait_until,            \
        unsigned short*:       shmem_ushort_wait_until,     \
        unsigned int*:         shmem_uint_wait_until,       \
        unsigned long*:        shmem_ulong_wait_until,      \
        unsigned long long*:   shmem_ulonglong_wait_until)(addr, cmp, value)
#endif

OSHMEM_DECLSPEC  int shmem_short_test(volatile short *addr, int cmp, short value);
OSHMEM_DECLSPEC  int shmem_int_test(volatile int *addr, int cmp, int value);
OSHMEM_DECLSPEC  int shmem_long_test(volatile long *addr, int cmp, long value);
OSHMEM_DECLSPEC  int shmem_longlong_test(volatile long long *addr, int cmp, long long value);
OSHMEM_DECLSPEC  int shmem_ushort_test(volatile unsigned short *addr, int cmp, unsigned short value);
OSHMEM_DECLSPEC  int shmem_uint_test(volatile unsigned int *addr, int cmp, unsigned int value);
OSHMEM_DECLSPEC  int shmem_ulong_test(volatile unsigned long *addr, int cmp, unsigned long value);
OSHMEM_DECLSPEC  int shmem_ulonglong_test(volatile unsigned long long *addr, int cmp, unsigned long long value);
OSHMEM_DECLSPEC  int shmem_int32_test(volatile int32_t *addr, int cmp, int32_t value);
OSHMEM_DECLSPEC  int shmem_int64_test(volatile int64_t *addr, int cmp, int64_t value);
OSHMEM_DECLSPEC  int shmem_uint32_test(volatile uint32_t *addr, int cmp, uint32_t value);
OSHMEM_DECLSPEC  int shmem_uint64_test(volatile uint64_t *addr, int cmp, uint64_t value);
OSHMEM_DECLSPEC  int shmem_size_test(volatile size_t *addr, int cmp, size_t value);
OSHMEM_DECLSPEC  int shmem_ptrdiff_test(volatile ptrdiff_t *addr, int cmp, ptrdiff_t value);
#if OSHMEM_HAVE_C11
#define shmem_test(addr, cmp, value)                  \
    _Generic(&*(addr),                                \
        short*:       shmem_short_test,               \
        int*:         shmem_int_test,                 \
        long*:        shmem_long_test,                \
        long long*:   shmem_longlong_test,            \
        unsigned short*:       shmem_ushort_test,     \
        unsigned int*:         shmem_uint_test,       \
        unsigned long*:        shmem_ulong_test,      \
        unsigned long long*:   shmem_ulonglong_test)(addr, cmp, value)
#endif

/*
 * Barrier sync routines
 */
OSHMEM_DECLSPEC  void shmem_barrier(int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC  void shmem_barrier_all(void);
OSHMEM_DECLSPEC  void shmem_sync(int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC  void shmem_sync_all(void);
OSHMEM_DECLSPEC  void shmem_fence(void);
OSHMEM_DECLSPEC  void shmem_ctx_fence(shmem_ctx_t ctx);
OSHMEM_DECLSPEC  void shmem_quiet(void);
OSHMEM_DECLSPEC  void shmem_ctx_quiet(shmem_ctx_t ctx);

/*
 * Collective routines
 */
OSHMEM_DECLSPEC void shmem_broadcast32(void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC void shmem_broadcast64(void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC void shmem_broadcast(void *target, const void *source, size_t nlong, int PE_root, int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC void shmem_collect32(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC void shmem_collect64(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC void shmem_fcollect32(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC void shmem_fcollect64(void *target, const void *source, size_t nlong, int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC void shmem_alltoall32(void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC void shmem_alltoall64(void *target, const void *source, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC void shmem_alltoalls32(void *target, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync);
OSHMEM_DECLSPEC void shmem_alltoalls64(void *target, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int PE_start, int logPE_stride, int PE_size, long *pSync);


/*
 * Reduction routines
 */
OSHMEM_DECLSPEC void shmem_short_and_to_all(short *target, const short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_int_and_to_all(int *target, const int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_long_and_to_all(long *target, const long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_longlong_and_to_all(long long *target, const long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmem_short_or_to_all(short *target, const short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_int_or_to_all(int *target, const int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_long_or_to_all(long *target, const long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_longlong_or_to_all(long long *target, const long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmem_short_xor_to_all(short *target, const short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_int_xor_to_all(int *target, const int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_long_xor_to_all(long *target, const long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_longlong_xor_to_all(long long *target, const long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmem_short_max_to_all(short *target, const short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_int_max_to_all(int *target, const int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_long_max_to_all(long *target, const long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_longlong_max_to_all(long long *target, const long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_float_max_to_all(float *target, const float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_double_max_to_all(double *target, const double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_longdouble_max_to_all(long double *target, const long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmem_short_min_to_all(short *target, const short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_int_min_to_all(int *target, const int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_long_min_to_all(long *target, const long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_longlong_min_to_all(long long *target, const long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_float_min_to_all(float *target, const float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_double_min_to_all(double *target, const double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_longdouble_min_to_all(long double *target, const long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmem_short_sum_to_all(short *target, const short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_int_sum_to_all(int *target, const int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_long_sum_to_all(long *target, const long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_longlong_sum_to_all(long long *target, const long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_float_sum_to_all(float *target, const float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_double_sum_to_all(double *target, const double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_longdouble_sum_to_all(long double *target, const long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_complexf_sum_to_all(OSHMEM_COMPLEX_TYPE(float) *target, const OSHMEM_COMPLEX_TYPE(float) *source, int nreduce, int PE_start, int logPE_stride, int PE_size, OSHMEM_COMPLEX_TYPE(float) *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_complexd_sum_to_all(OSHMEM_COMPLEX_TYPE(double) *target, const OSHMEM_COMPLEX_TYPE(double) *source, int nreduce, int PE_start, int logPE_stride, int PE_size, OSHMEM_COMPLEX_TYPE(double) *pWrk, long *pSync);

OSHMEM_DECLSPEC void shmem_short_prod_to_all(short *target, const short *source, int nreduce, int PE_start, int logPE_stride, int PE_size, short *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_int_prod_to_all(int *target, const int *source, int nreduce, int PE_start, int logPE_stride, int PE_size, int *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_long_prod_to_all(long *target, const long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_longlong_prod_to_all(long long *target, const long long *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long long *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_float_prod_to_all(float *target, const float *source, int nreduce, int PE_start, int logPE_stride, int PE_size, float *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_double_prod_to_all(double *target, const double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, double *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_longdouble_prod_to_all(long double *target, const long double *source, int nreduce, int PE_start, int logPE_stride, int PE_size, long double *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_complexf_prod_to_all(OSHMEM_COMPLEX_TYPE(float) *target, const OSHMEM_COMPLEX_TYPE(float) *source, int nreduce, int PE_start, int logPE_stride, int PE_size, OSHMEM_COMPLEX_TYPE(float) *pWrk, long *pSync);
OSHMEM_DECLSPEC void shmem_complexd_prod_to_all(OSHMEM_COMPLEX_TYPE(double) *target, const OSHMEM_COMPLEX_TYPE(double) *source, int nreduce, int PE_start, int logPE_stride, int PE_size, OSHMEM_COMPLEX_TYPE(double) *pWrk, long *pSync);

/*
 * Platform specific cache management routines
 */
OSHMEM_DECLSPEC void shmem_udcflush(void);
OSHMEM_DECLSPEC void shmem_udcflush_line(void* target);
OSHMEM_DECLSPEC void shmem_set_cache_inv(void);
OSHMEM_DECLSPEC void shmem_set_cache_line_inv(void* target);
OSHMEM_DECLSPEC void shmem_clear_cache_inv(void);
OSHMEM_DECLSPEC void shmem_clear_cache_line_inv(void* target);

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif


#endif /* OSHMEM_SHMEM_H */
