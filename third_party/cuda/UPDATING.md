These headers come from CUDA & CUPTI SDKs.

From CUDA:
- `cuda.h`
- `cuda_stdint.h`

From CUPTI:
- `cupti_activity.h`
- `cupti_callbacks.h`
- `cupti_driver_cbid.h`
- `cupti_events.h`
- `cupti_metrics.h`
- `cupti_result.h`
- `cupti_runtime_cbid.h`
- `cupti.h`

## CUDA SDK Update

To update, install CUDA SDK locally:
```
sudo apt-get install cuda
```

Copy cuda headers, version.txt and libdevice.10.bc:
```
cp /usr/local/cuda/include/cuda.h ./include/
cp /usr/local/cuda/include/cuda_stdint.h ./include/
cp /usr/local/cuda/version.txt  .
cp /usr/local/cuda/nvvm/libdevice/libdevice.10.bc ./nvvm/libdevice/
```

## CUPTI SDK Update

To update, install CUDA SDK locally:
```
sudo apt-get install libcupti-dev
```

Copy cupti headers:
```
cp /usr/local/cuda/extras/CUPTI/include/cupti_activity.h ./include/
cp /usr/local/cuda/extras/CUPTI/include/cupti_callbacks.h ./include/
cp /usr/local/cuda/extras/CUPTI/include/cupti_driver_cbid.h ./include/
cp /usr/local/cuda/extras/CUPTI/include/cupti_events.h ./include/
cp /usr/local/cuda/extras/CUPTI/include/cupti_metrics.h ./include/
cp /usr/local/cuda/extras/CUPTI/include/cupti_result.h ./include/
cp /usr/local/cuda/extras/CUPTI/include/cupti_runtime_cbid.h ./include/
```

### Modifications to CUPTI

Note: `cupti.h` is not copied/updated; we use a truncated version that simply lists the aforementioned headers.

From `cupti_callbacks.h`, comment out:
```
#include <builtin_types.h>
```
This is not required/included for building IREE CUPTI targets.
