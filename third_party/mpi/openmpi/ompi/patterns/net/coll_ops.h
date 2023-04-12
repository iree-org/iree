/*
 * Copyright (c) 2009-2012 Mellanox Technologies.  All rights reserved.
 * Copyright (c) 2009-2012 Oak Ridge National Laboratory.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef COMM_OP_TYPES_H
#define COMM_OP_TYPES_H

#include "ompi_config.h"

BEGIN_C_DECLS

int comm_allreduce(void *sbuf, void *rbuf, int count, opal_datatype_t *dtype,
                int op, opal_list_t *peers);

/* reduction operations supported */
#define OP_SUM 1

#define TYPE_INT4 1


END_C_DECLS

#endif /* COMM_OP_TYPES_H */
