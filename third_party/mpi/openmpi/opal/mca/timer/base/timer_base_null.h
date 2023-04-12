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

#ifndef OPAL_MCA_TIMER_BASE_TIMER_BASE_NULL_H
#define OPAL_MCA_TIMER_BASE_TIMER_BASE_NULL_H

typedef int opal_timer_t;

static inline opal_timer_t
opal_timer_base_get_cycles(void)
{
    return 0;
}


static inline opal_timer_t
opal_timer_base_get_usec(void)
{
    return 0;
}


static inline opal_timer_t
opal_timer_base_get_freq(void)
{
    return 0;
}


#define OPAL_TIMER_CYCLE_NATIVE 0
#define OPAL_TIMER_CYCLE_SUPPORTED 0
#define OPAL_TIMER_USEC_NATIVE 0
#define OPAL_TIMER_USEC_SUPPORTED 0

#endif
