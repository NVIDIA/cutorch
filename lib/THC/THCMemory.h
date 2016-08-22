#ifndef THC_MEMORY_INC
#define THC_MEMORY_INC

#include "THCGeneral.h"

typedef enum
  {
    /* use cudaMalloc/Free directly */
    THC_MEMORY_POOL_MODE_NONE = 0,
    /* utilize CNMEM memory pool */
    THC_MEMORY_POOL_MODE_CNMEM = 1
  } THCudaMemoryPoolMode;

/* keys to initialize memory pool on cutorch load from the environment */
#define THC_MEMORY_POOL_ENV_KEY_MODE             "CUTORCH_GPU_MEMORY_POOL_MODE"
#define THC_MEMORY_POOL_ENV_VAL_MODE_CNMEM       "CNMEM"
#define THC_MEMORY_POOL_ENV_KEY_PERCENT          "CUTORCH_GPU_MEMORY_POOL_PERCENT"
/* Default initial memory pool size is 20% of free memory */
#define THC_MEMORY_POOL_ENV_VAL_PERCENT_DEFAULT  20

THC_API void THCudaMemoryInit(THCState* state);

/* Straightforward call to cudaMalloc or pool allocation function */
THC_API cudaError_t THCudaTryMalloc(THCState *state, void** ptr, size_t size);

/* this one will THCudaTryMalloc, and in OOM case will run GC hook and retry allocation */
THC_API cudaError_t THCudaMalloc(THCState *state, void **ptr, size_t size);

THC_API cudaError_t THCudaFree(THCState *state, void *ptr);

THC_API cudaError_t THCudaMemoryAllocateLargestBlock(THCState* state, int device, void** ptr, size_t* size);

THC_API void THCudaMemoryGetInfo(THCState* state, int device, size_t* freeBytes, size_t* totalBytes);

THC_API void THCudaMemoryPoolActivate(THCState *state, THCudaMemoryPoolMode mode);

THC_API void THCudaMemoryPoolDeactivate(THCState* state);

THC_API void THCudaMemoryPoolRegisterStream(THCState *state, cudaStream_t stream);

THC_API void THCSetGCHandler(THCState *state,
                             void (*torchGCHandlerFunction)(void *data),
                             void *data );
THC_API void THCHeapUpdate(THCState *state, long size);

#endif
