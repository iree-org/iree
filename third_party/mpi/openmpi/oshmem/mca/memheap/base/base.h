/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/**
 * @file
 */
#ifndef MCA_MEMHEAP_BASE_H
#define MCA_MEMHEAP_BASE_H

#include "oshmem_config.h"
#include "opal/class/opal_list.h"
#include "opal/class/opal_value_array.h"
#include "oshmem/mca/mca.h"

#include "oshmem/mca/memheap/memheap.h"

BEGIN_C_DECLS

/*
 * Global functions for MCA: overall MEMHEAP open and close
 */
OSHMEM_DECLSPEC int mca_memheap_base_select(void);

/*
 * Globals
 */

/* only used within base -- no need to DECLSPEC */
#define MEMHEAP_BASE_MIN_ORDER         3                                /* forces 64 bit alignment */
#define MEMHEAP_BASE_PAGE_ORDER        21
#define MEMHEAP_BASE_PRIVATE_SIZE      (1ULL << MEMHEAP_BASE_PAGE_ORDER) /* should be at least the same as a huge page size */
#define MEMHEAP_BASE_MIN_SIZE          (1ULL << MEMHEAP_BASE_PAGE_ORDER) /* must fit into at least one huge page */

extern int mca_memheap_base_already_opened;
extern int mca_memheap_base_key_exchange;

#define MCA_MEMHEAP_MAX_SEGMENTS    32
#define HEAP_SEG_INDEX              0
#define MCA_MEMHEAP_SEG_COUNT       2

#define MEMHEAP_SEG_INVALID  0xFFFF


typedef struct mca_memheap_base_config {
    long            device_nic_mem_seg_size; /* Used for SHMEM_HINT_DEVICE_NIC_MEM */
} mca_memheap_base_config_t;


typedef struct mca_memheap_map {
    map_segment_t   mem_segs[MCA_MEMHEAP_MAX_SEGMENTS]; /* TODO: change into pointer array */
    int             n_segments;
    int             num_transports;
} mca_memheap_map_t;

extern mca_memheap_map_t mca_memheap_base_map;
extern mca_memheap_base_config_t mca_memheap_base_config;

int mca_memheap_base_alloc_init(mca_memheap_map_t *, size_t, long, char *);
void mca_memheap_base_alloc_exit(mca_memheap_map_t *);
int mca_memheap_base_static_init(mca_memheap_map_t *);
void mca_memheap_base_static_exit(mca_memheap_map_t *);
int mca_memheap_base_reg(mca_memheap_map_t *);
int mca_memheap_base_dereg(mca_memheap_map_t *);
int memheap_oob_init(mca_memheap_map_t *);
void memheap_oob_destruct(void);

OSHMEM_DECLSPEC int mca_memheap_base_is_symmetric_addr(const void* va);
OSHMEM_DECLSPEC sshmem_mkey_t *mca_memheap_base_get_mkey(void* va,
                                                           int tr_id);
OSHMEM_DECLSPEC sshmem_mkey_t * mca_memheap_base_get_cached_mkey_slow(shmem_ctx_t ctx,
                                                                      map_segment_t *s,
                                                                      int pe,
                                                                      void* va,
                                                                      int btl_id,
                                                                      void** rva);
OSHMEM_DECLSPEC void mca_memheap_modex_recv_all(void);

/* This function is for internal usage only
 * return value:
 * 0 - addr is not symmetric address
 * 1 - addr is part of user memheap
 * 2 - addr is part of private memheap
 * 3 - addr is static variable
 */
typedef enum {
    ADDR_INVALID = 0, ADDR_USER, ADDR_PRIVATE, ADDR_STATIC,
} addr_type_t;

OSHMEM_DECLSPEC int mca_memheap_base_detect_addr_type(void* va);

static inline unsigned memheap_log2(unsigned long long val)
{
    /* add 1 if val is NOT a power of 2 (to do the ceil) */
    unsigned int count = (val & (val - 1) ? 1 : 0);

    while (val > 0) {
        val = val >> 1;
        count++;
    }

    return count > 0 ? count - 1 : 0;
}

static inline void *memheap_down_align_addr(void* addr, unsigned int shift)
{
    return (void*) (((intptr_t) addr) & (~(intptr_t) 0) << shift);
}

static inline void *memheap_up_align_addr(void*addr, unsigned int shift)
{
    return (void*) ((((intptr_t) addr) | ~((~(intptr_t) 0) << shift)));
}

static inline unsigned long long memheap_align(unsigned long top)
{
    return ((top + MEMHEAP_BASE_MIN_SIZE - 1) & ~(MEMHEAP_BASE_MIN_SIZE - 1));
}

/*
 * MCA framework
 */
OSHMEM_DECLSPEC extern mca_base_framework_t oshmem_memheap_base_framework;

/* ******************************************************************** */
#ifdef __BASE_FILE__
#define __SPML_FILE__ __BASE_FILE__
#else
#define __SPML_FILE__ __FILE__
#endif

#ifdef OPAL_ENABLE_DEBUG
#define MEMHEAP_VERBOSE(level, ...) \
    oshmem_output_verbose(level, oshmem_memheap_base_framework.framework_output, \
        "%s:%d - %s()", __SPML_FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define MEMHEAP_VERBOSE(level, ...)
#endif

#define MEMHEAP_ERROR(...) \
    oshmem_output(oshmem_memheap_base_framework.framework_output, \
        "Error %s:%d - %s()", __SPML_FILE__, __LINE__, __func__, __VA_ARGS__)

#define MEMHEAP_WARN(...) \
    oshmem_output_verbose(0, oshmem_memheap_base_framework.framework_output, \
        "Warning %s:%d - %s()", __SPML_FILE__, __LINE__, __func__, __VA_ARGS__)

extern int mca_memheap_seg_cmp(const void *k, const void *v);

/* Turn ON/OFF debug output from build (default 0) */
#ifndef MEMHEAP_BASE_DEBUG
#define MEMHEAP_BASE_DEBUG    0
#endif
#define MEMHEAP_VERBOSE_FASTPATH(...)

extern mca_memheap_map_t* memheap_map;

static inline int map_segment_is_va_in(map_base_segment_t *s, void *va)
{
    return (va >= s->va_base && va < s->va_end);
}

static inline map_segment_t *memheap_find_seg(int segno)
{
    return &mca_memheap_base_map.mem_segs[segno];
}

static inline int memheap_is_va_in_segment(void *va, int segno) 
{
    return map_segment_is_va_in(&memheap_find_seg(segno)->super, va);
}

static inline int memheap_find_segnum(void *va, int pe)
{
    int i;
    int my_pe = oshmem_my_proc_id();

    if (pe == my_pe) {
        /* Find segment number for local segment using va_base
         * TODO: Merge local and remote segment information in mkeys_cache
         */
        for (i = 0; i < mca_memheap_base_map.n_segments; i++) {
            if (memheap_is_va_in_segment(va, i)) {
                return i;
            }
        }
    } else {
        /* Find segment number for remote segments using va_base */
        for (i = 0; i < mca_memheap_base_map.n_segments; i++) {
            map_segment_t *seg = memheap_find_seg(i);
            if (seg) {
                sshmem_mkey_t **mkeys_cache = seg->mkeys_cache;
                if (mkeys_cache) {
                    if (mkeys_cache[pe]) {
                        if ((va >= mkeys_cache[pe]->va_base) &&
                            ((char*)va < (char*)mkeys_cache[pe]->va_base + mkeys_cache[pe]->len)) {
                            return i;
                        }
                    }
                }
            }
        }
    }
    return MEMHEAP_SEG_INVALID;
}

static inline void* memheap_va2rva(void* va, void* local_base, void* remote_base)
{
    return (void*) (remote_base > local_base ?
            (uintptr_t)va + ((uintptr_t)remote_base - (uintptr_t)local_base) :
            (uintptr_t)va - ((uintptr_t)local_base - (uintptr_t)remote_base));
}

static inline void *map_segment_va2rva(mkey_segment_t *seg, void *va)
{
    return memheap_va2rva(va, seg->super.va_base, seg->rva_base);
}

void mkey_segment_init(mkey_segment_t *seg, sshmem_mkey_t *mkey, uint32_t segno);

static inline map_segment_t *memheap_find_va(void* va)
{
    map_segment_t *s = NULL;
    int i;

    for (i = 0; i < memheap_map->n_segments; i++) {
        if (memheap_is_va_in_segment(va, i)) {
            s = &memheap_map->mem_segs[i];
            break;
        }
    }

#if MEMHEAP_BASE_DEBUG == 1
    if (s) {
        MEMHEAP_VERBOSE(5, "match seg#%02ld: 0x%llX - 0x%llX %llu bytes va=%p",
                s - memheap_map->mem_segs,
                (long long)s->super.va_base,
                (long long)s->super.va_end,
                (long long)(s->super.va_end - s->super.va_base),
                (void *)va);
    }
#endif
    return s;
}

static inline  sshmem_mkey_t *mca_memheap_base_get_cached_mkey(shmem_ctx_t ctx,
                                                               int pe,
                                                                void* va,
                                                                int btl_id,
                                                                void** rva)
{
    map_segment_t *s;
    sshmem_mkey_t *mkey;

    MEMHEAP_VERBOSE_FASTPATH(10, "rkey: pe=%d va=%p", pe, va);
    s = memheap_find_va(va);
    if (OPAL_UNLIKELY(NULL == s))
        return NULL ;

    if (OPAL_UNLIKELY(!MAP_SEGMENT_IS_VALID(s)))
        return NULL ;

    if (OPAL_UNLIKELY(pe == oshmem_my_proc_id())) {
        *rva = va;
        MEMHEAP_VERBOSE_FASTPATH(10, "rkey: pe=%d va=%p -> (local) %lx %p", pe, va, 
                s->mkeys[btl_id].u.key, *rva);
        return &s->mkeys[btl_id];
    }

    if (OPAL_LIKELY(s->mkeys_cache[pe])) {
        mkey = &s->mkeys_cache[pe][btl_id];
        *rva = memheap_va2rva(va, s->super.va_base, mkey->va_base);
        MEMHEAP_VERBOSE_FASTPATH(10, "rkey: pe=%d va=%p -> (cached) %lx %p", pe, (void *)va, mkey->u.key, (void *)*rva);
        return mkey;
    }

    return mca_memheap_base_get_cached_mkey_slow(ctx, s, pe, va, btl_id, rva);
}

static inline int mca_memheap_base_num_transports(void) 
{
    return memheap_map->num_transports;
}

static inline void* mca_memheap_seg2base_va(int seg)
{
    return memheap_map->mem_segs[seg].super.va_base;
}

END_C_DECLS

#endif /* MCA_MEMHEAP_BASE_H */
