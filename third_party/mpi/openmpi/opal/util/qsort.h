/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_QSORT_H
#define OPAL_QSORT_H

#if OPAL_HAVE_BROKEN_QSORT

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h> /* for size_t */
#endif

BEGIN_C_DECLS

void opal_qsort(void *a, size_t n, size_t es, int (*cmp)(const void *, const void*));

END_C_DECLS

#else
#error "Don't include opal/qsort/qsort.h directly"
#endif /* OPAL_HAVE_BROKEN_QSORT */

#endif
