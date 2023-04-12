/*
 * Copyright (c) 2004-2006 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_MCA_MEMCPY_BASE_MEMCPY_BASE_NULL_H
#define OPAL_MCA_MEMCPY_BASE_MEMCPY_BASE_NULL_H

#define opal_memcpy( dst, src, length ) \
    memcpy( (dst), (src), (length) );

#define opal_memcpy_tov( dst_iov, src, count )        \
    do {                                              \
        int _i;                                       \
        char* _src = (char*)src;                      \
                                                      \
        for( _i = 0; _i < count; _i++ ) {             \
            opal_memcpy( dst_iov[_i].iov_base, _src,  \
                         dst_iov[_i].iov_len );       \
            _src += dst_iov[_i].iov_len;              \
        }                                             \
    } while (0)

#define opal_memcpy_fromv( dst, src_iov, count )        \
    do {                                                \
        int _i;                                         \
        char* _dst = (char*)dst;                        \
                                                        \
        for( _i = 0; _i < count; _i++ ) {               \
            opal_memcpy( _dst, src_iov[_i].iov_base,    \
                         src_iov[_i].iov_len );         \
            _dst += src_iov[_i].iov_len;                \
        }                                               \
    } while (0)

#endif
