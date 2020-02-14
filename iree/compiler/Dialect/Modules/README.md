The dialects in this directory model the runtime modules in `iree/modules/`. For
each function in the runtime module, there should be a corresponding op that
maps 1:1 with it.

TODO(silvasean): Move the `iree/modules/*/dialect` directories into here.

The HAL module is an exception to this, due to its central importance in the
IREE compiler flow.
