/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2011      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2017-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_DATATYPE_UNPACK_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_UNPACK_H_HAS_BEEN_INCLUDED

#include "opal_config.h"

#if !defined(CHECKSUM) && OPAL_CUDA_SUPPORT
/* Make use of existing macro to do CUDA style memcpy */
#undef MEMCPY_CSUM
#define MEMCPY_CSUM( DST, SRC, BLENGTH, CONVERTOR ) \
    CONVERTOR->cbmemcpy( (DST), (SRC), (BLENGTH), (CONVERTOR) )
#endif

/**
 * This function deals only with partial elements. The COUNT points however to
 * the whole leftover count, but this function is only expected to operate on
 * an amount less than blength, that would allow the rest of the pack process
 * to handle only entire blength blocks (plus the left over).
 *
 * Return 1 if we are now aligned on a block, 0 otherwise.
 */
static inline int
unpack_partial_blocklen( opal_convertor_t* CONVERTOR,
                        const dt_elem_desc_t* ELEM,
                        size_t* COUNT,
                        unsigned char** packed,
                        unsigned char** memory,
                        size_t* SPACE )
{
    const ddt_elem_desc_t* _elem = &((ELEM)->elem);
    size_t do_now_bytes = opal_datatype_basicDatatypes[_elem->common.type]->size;
    size_t do_now = (*COUNT);
    unsigned char* _memory = (*memory) + _elem->disp;
    unsigned char* _packed = *packed;

    assert( *(COUNT) <= ((size_t)(_elem->count * _elem->blocklen)) );

    if( (*SPACE) < do_now_bytes ) /* Can we do anything ? */
        return 0;
    /**
     * First check if we already did something on this element ? The COUNT is the number
     * of remaining predefined types in the current elem, not how many predefined types
     * should be manipulated in the current call (this number is instead reflected on the
     * SPACE).
     */
    if( 0 == (do_now = (*COUNT) % _elem->blocklen) )
        return 1;

    size_t left_in_block = do_now;  /* left in the current blocklen */

    if( (do_now_bytes * do_now) > *(SPACE) )
        do_now = (*SPACE) / do_now_bytes;

    do_now_bytes *= do_now;

    OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, do_now_bytes, (CONVERTOR)->pBaseBuf,
                                     (CONVERTOR)->pDesc, (CONVERTOR)->count );
    DO_DEBUG( opal_output( 0, "unpack memcpy( %p [%ld], %p, %lu ) => space %lu [prolog]\n",
                           (void*)_memory, _memory - CONVERTOR->pBaseBuf,
                           (void*)_packed, (unsigned long)do_now_bytes, (unsigned long)(*(SPACE)) ); );
    MEMCPY_CSUM( _memory, _packed, do_now_bytes, (CONVERTOR) );
    *(memory)     += (ptrdiff_t)do_now_bytes;
    if( do_now == left_in_block )  /* compensate if completed a blocklen */
        *(memory) += _elem->extent - (_elem->blocklen * opal_datatype_basicDatatypes[_elem->common.type]->size);

    *(COUNT)  -= do_now;
    *(SPACE)  -= do_now_bytes;
    *(packed) += do_now_bytes;
    return (do_now == left_in_block);
}

static inline void
unpack_predefined_data( opal_convertor_t* CONVERTOR,
                        const dt_elem_desc_t* ELEM,
                        size_t* COUNT,
                        unsigned char** packed,
                        unsigned char** memory,
                        size_t* SPACE )
{
    const ddt_elem_desc_t* _elem = &((ELEM)->elem);
    size_t blocklen_bytes = opal_datatype_basicDatatypes[_elem->common.type]->size;
    size_t cando_count = (*COUNT), do_now_bytes;
    unsigned char* _memory = (*memory) + _elem->disp;
    unsigned char* _packed = *packed;

    assert( 0 == (cando_count % _elem->blocklen) );  /* no partials here */
    assert( *(COUNT) <= ((size_t)_elem->count * _elem->blocklen) );

    if( (blocklen_bytes * cando_count) > *(SPACE) )
        cando_count = (*SPACE) / blocklen_bytes;

    /* preemptively update the number of COUNT we will return. */
    *(COUNT) -= cando_count;

    if (1 == _elem->blocklen) {  /* Do as many full blocklen as possible */
        for (; cando_count > 0; cando_count--) {
            OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, blocklen_bytes, (CONVERTOR)->pBaseBuf,
                                             (CONVERTOR)->pDesc, (CONVERTOR)->count );
            DO_DEBUG( opal_output( 0, "unpack memcpy( %p [%ld], %p [%ld], %lu ) => space %lu [blen = 1]\n",
                                   (void*)_memory, _memory - CONVERTOR->pBaseBuf,
                                   (void*)_packed, _packed - *packed,
                                   (unsigned long)blocklen_bytes, (unsigned long)(*(SPACE) - (_packed - *(packed))) ); );
            MEMCPY_CSUM( _memory, _packed, blocklen_bytes, (CONVERTOR) );
            _packed     += blocklen_bytes;
            _memory     += _elem->extent;
        }
        goto update_and_return;
    }

    if( (1 < _elem->count) && (_elem->blocklen <= cando_count) ) {
        blocklen_bytes *= _elem->blocklen;

        do { /* Do as many full blocklen as possible */
            OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, blocklen_bytes, (CONVERTOR)->pBaseBuf,
                                             (CONVERTOR)->pDesc, (CONVERTOR)->count );
            DO_DEBUG( opal_output( 0, "unpack 2. memcpy( %p [%ld], %p [%ld], %lu ) => space %lu\n",
                                   (void*)_memory, _memory - CONVERTOR->pBaseBuf,
                                   (void*)_packed, _packed - *packed,
                                   (unsigned long)blocklen_bytes, (unsigned long)(*(SPACE) - (_packed - *(packed))) ); );
            MEMCPY_CSUM( _memory, _packed, blocklen_bytes, (CONVERTOR) );
            _packed     += blocklen_bytes;
            _memory     += _elem->extent;
            cando_count -= _elem->blocklen;
        } while (_elem->blocklen <= cando_count);
    }

    /**
     * As an epilog do anything left from the last blocklen.
     */
    if( 0 != cando_count ) {
        assert( (cando_count < _elem->blocklen) ||
                ((1 == _elem->count) && (cando_count <= _elem->blocklen)) );
        do_now_bytes = cando_count * opal_datatype_basicDatatypes[_elem->common.type]->size;
        OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, do_now_bytes, (CONVERTOR)->pBaseBuf,
                                         (CONVERTOR)->pDesc, (CONVERTOR)->count );
        DO_DEBUG( opal_output( 0, "unpack 3. memcpy( %p [%ld], %p [%ld], %lu ) => space %lu [epilog]\n",
                               (void*)_memory, _memory - CONVERTOR->pBaseBuf,
                               (void*)_packed, _packed - *packed,
                               (unsigned long)do_now_bytes, (unsigned long)(*(SPACE) - (_packed - *(packed))) ); );
        MEMCPY_CSUM( _memory, _packed, do_now_bytes, (CONVERTOR) );
        _memory   += do_now_bytes;
        _packed   += do_now_bytes;
    }

 update_and_return:
    *(memory)  = _memory - _elem->disp;
    *(SPACE)  -= (_packed - *packed);
    *(packed)  = _packed;
}

static inline void unpack_contiguous_loop( opal_convertor_t* CONVERTOR,
                                           const dt_elem_desc_t* ELEM,
                                           size_t* COUNT,
                                           unsigned char** packed,
                                           unsigned char** memory,
                                           size_t* SPACE )
{
    const ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    const ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _memory = (*memory) + _end_loop->first_elem_disp;
    size_t _copy_loops = *(COUNT);

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (*(SPACE) / _end_loop->size);
    for(size_t _i = 0; _i < _copy_loops; _i++ ) {
        OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, _end_loop->size, (CONVERTOR)->pBaseBuf,
                                    (CONVERTOR)->pDesc, (CONVERTOR)->count );
        DO_DEBUG( opal_output( 0, "unpack 3. memcpy( %p, %p, %lu ) => space %lu\n",
                               (void*)_memory, (void*)*(packed), (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _i * _end_loop->size) ); );
        MEMCPY_CSUM( _memory, *(packed), _end_loop->size, (CONVERTOR) );
        *(packed) += _end_loop->size;
        _memory   += _loop->extent;
    }
    *(memory)  = _memory - _end_loop->first_elem_disp;
    *(SPACE)  -= _copy_loops * _end_loop->size;
    *(COUNT)  -= _copy_loops;
}

#define UNPACK_PARTIAL_BLOCKLEN( CONVERTOR,    /* the convertor */       \
                                 ELEM,         /* the basic element to be packed */ \
                                 COUNT,        /* the number of elements */ \
                                 PACKED,       /* the destination pointer (char*) */ \
                                 MEMORY,       /* the source pointer (char*) */ \
                                 SPACE )       /* the space in the destination buffer */ \
unpack_partial_blocklen( (CONVERTOR), (ELEM), &(COUNT), &(PACKED), &(MEMORY), &(SPACE) )

#define UNPACK_PREDEFINED_DATATYPE( CONVERTOR,    /* the convertor */    \
                                    ELEM,         /* the basic element to be packed */ \
                                    COUNT,        /* the number of elements */ \
                                    PACKED,       /* the destination pointer (char*) */ \
                                    MEMORY,       /* the source pointer (char*) */ \
                                    SPACE )       /* the space in the destination buffer */ \
unpack_predefined_data( (CONVERTOR), (ELEM), &(COUNT), &(PACKED), &(MEMORY), &(SPACE) )

#define UNPACK_CONTIGUOUS_LOOP( CONVERTOR, ELEM, COUNT, PACKED, MEMORY, SPACE ) \
    unpack_contiguous_loop( (CONVERTOR), (ELEM), &(COUNT), &(PACKED), &(MEMORY), &(SPACE) )

#endif  /* OPAL_DATATYPE_UNPACK_H_HAS_BEEN_INCLUDED */
