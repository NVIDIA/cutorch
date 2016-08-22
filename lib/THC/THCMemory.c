#include "THCAllocator.h"
#include "THCMemory.h"
#include "TH.h"

#include "cnmem.h"


/*******************   Memory Pool Stuff   **********************************/

static cudaError_t THC_cnmem2cuda(cnmemStatus_t err) {
  switch (err) {
  case CNMEM_STATUS_SUCCESS:
    return cudaSuccess;
  case CNMEM_STATUS_OUT_OF_MEMORY:
    return cudaErrorMemoryAllocation;
  case CNMEM_STATUS_NOT_INITIALIZED:
    return cudaErrorInitializationError;
  case CNMEM_STATUS_CUDA_ERROR:
    return cudaGetLastError();
  default:
    return cudaErrorUnknown;
  }
}

cudaError_t THCudaTryMalloc(THCState *state, void** ptr, size_t size)
{
  cudaError_t err = cudaSuccess;
  cudaGetLastError(); // reset OOM error
  switch (state->memoryPoolMode) {
  case THC_MEMORY_POOL_MODE_CNMEM:
    err = THC_cnmem2cuda(cnmemMalloc(ptr, size, THCState_getCurrentStream(state)));
    break;
  default:
    err = cudaMalloc(ptr, size);
  }
  return err;
}

// statistic
static int numAllocations = 0;
static int numDeallocations = 0;

cudaError_t THCudaMalloc(THCState *state, void** ptr, size_t size)
{
  cudaError_t err = THCudaTryMalloc(state, ptr, size);
  if (state->cutorchGCFunction != NULL && err != cudaSuccess) {
     cudaGetLastError(); // reset OOM error
     (state->cutorchGCFunction)(state->cutorchGCData);
     err = THCudaTryMalloc(state, ptr, size);
  }
  ++numAllocations;
  THMemoryCheck(err);
  return err;
}

cudaError_t THCudaFree(THCState *state, void *ptr)
{
  cudaError_t err = cudaSuccess;
  ++numDeallocations;
  switch (state->memoryPoolMode) {
  case THC_MEMORY_POOL_MODE_CNMEM:
    {
    cnmemStatus_t cnerr = cnmemFree(ptr, THCState_getCurrentStream(state));
    if (cnerr==CNMEM_STATUS_INVALID_ARGUMENT)
      /* this usually means cnmem have not found it in its list */
      err = cudaFree(ptr);
    else
      err = THC_cnmem2cuda(cnerr);
    }
    break;
  default:
    err = cudaFree(ptr);
  }
  return err;
}


void THCudaMemoryGetInfo_aux(THCState* state, int device, size_t* freeBytes, size_t* totalBytes, size_t* largest)
{
  int curDevice = device;
  cudaStream_t stream = NULL;
  double fragmentationGuess = 0.9;

  if (device >=0) {
    THCudaCheck(cudaGetDevice(&curDevice));
    THCudaCheck(cudaSetDevice(device));
  }
  switch (state->memoryPoolMode) {
  case THC_MEMORY_POOL_MODE_CNMEM:
    stream = THCState_getCurrentStream(state);
    THCudaCheck(cnmemMemGetInfo(freeBytes, totalBytes, stream));
    if (largest) {
      // todo: implement a special method to find largest free block in the pool
      // for now, hope for little fragmentation
      size_t largestInThePool=(*freeBytes)*fragmentationGuess;
      size_t freeBytesOut, totalBytesOut;
      THCudaCheck(cudaMemGetInfo(&freeBytesOut, &totalBytesOut));
      // compare largest free block in the pool with the outside free space
      *largest = (freeBytesOut > largestInThePool ? freeBytesOut*fragmentationGuess : largestInThePool);
    }
    break;
  default:
    THCudaCheck(cudaMemGetInfo(freeBytes, totalBytes));
    if (largest)
      *largest = (*freeBytes) * fragmentationGuess;
    break;
  }
  if (device != curDevice)
    THCudaCheck(cudaSetDevice(curDevice));
}

void THCudaMemoryGetInfo(THCState* state, int device, size_t* freeBytes, size_t* totalBytes)
{
  THCudaMemoryGetInfo_aux(state, device, freeBytes, totalBytes, NULL);
}

static THCudaMemoryPoolMode THC_parseMemoryPoolMode(const char* mode)
{
  if (!strcmp(mode, THC_MEMORY_POOL_ENV_VAL_MODE_CNMEM))
    return THC_MEMORY_POOL_MODE_CNMEM;
  else
    return THC_MEMORY_POOL_MODE_NONE;
}

cudaError_t THCudaMemoryAllocateLargestBlock(THCState* state, int device, void** ptr, size_t* size)
{
  size_t freeBytes, totalBytes, largest;
  THCudaMemoryGetInfo_aux(state, device, &freeBytes, &totalBytes, &largest);
  cudaError_t err = THCudaTryMalloc(state, ptr, largest);
  int tries = 0;
  while (err != cudaSuccess && tries++ < 10 ) {
    largest = (largest/10)*8;
    if (state->cutorchGCFunction != NULL) {
      (state->cutorchGCFunction)(state->cutorchGCData);
    }
    err = THCudaTryMalloc(state, ptr, largest);
  }
  THCudaCheck(err);
  *size = largest;
  return err;
}

static void THC_cnmem_init(THCState* state) {
  cnmemDevice_t devices[state->numDevices];

  for (int i = 0; i < state->numDevices ; ++i) {
    size_t free_mem, total_mem;
    size_t percent = state->memoryPoolPercent < 0 ? THC_MEMORY_POOL_ENV_VAL_PERCENT_DEFAULT :
      (state->memoryPoolPercent > 100 ? 100 : state->memoryPoolPercent);
    cudaSetDevice(i);
    devices[i].device = i;
    cudaMemGetInfo(&free_mem, &total_mem);
    devices[i].size = (free_mem/100)*state->memoryPoolPercent;

    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, i);
    devices[i].numStreams = state->numUserStreams;
    devices[i].streams = res->streams;
    /* not sure about those 'reserve' sizes */
    devices[i].streamSizes = NULL;
  }

  THCudaCheck(THC_cnmem2cuda(cnmemInit(state->numDevices, devices, CNMEM_FLAGS_DEFAULT)));
  if (state->verbose)
    cnmemPrintMemoryState(stderr, THCState_getCurrentStream(state));
}

void THCudaMemoryInit(THCState* state)
{
  /* initialize memory pool settings */
  state->memoryPoolMode = THC_MEMORY_POOL_MODE_NONE;
  const char* poolMode = getenv(THC_MEMORY_POOL_ENV_KEY_MODE);
  const char* poolPercent = getenv(THC_MEMORY_POOL_ENV_KEY_PERCENT);
  state->memoryPoolPercent = poolPercent ? atoi(poolPercent) : THC_MEMORY_POOL_ENV_VAL_PERCENT_DEFAULT;
  if (poolMode)
    THCudaMemoryPoolActivate(state, THC_parseMemoryPoolMode(poolMode));
}

void THCudaMemoryPoolDeactivate(THCState* state)
{
  if (state->verbose)
    fprintf(stderr, "THCudaMemoryPoolDeactivate,  allocations : %d deallocations: %d\n",
            numAllocations, numDeallocations);
  if (state->memoryPoolMode != THC_MEMORY_POOL_MODE_NONE)
    THCudaMemoryPoolActivate(state,THC_MEMORY_POOL_MODE_NONE);
}


void THCudaMemoryPoolActivate(THCState *state, THCudaMemoryPoolMode mode)
{
  if (state->memoryPoolMode != mode) {
    switch (state->memoryPoolMode) {
    case THC_MEMORY_POOL_MODE_CNMEM:
      cnmemFinalize();
      break;
    default:
      break;
    }
    state->memoryPoolMode = THC_MEMORY_POOL_MODE_NONE;
    switch (mode) {
    case THC_MEMORY_POOL_MODE_CNMEM:
      THC_cnmem_init(state);
      state->memoryPoolMode = mode;
      break;
    default:
      break;
    }
  }
}

void THCudaMemoryPoolRegisterStream(THCState *state, cudaStream_t stream)
{
  switch (state->memoryPoolMode) {
  case THC_MEMORY_POOL_MODE_CNMEM:
    cnmemRegisterStream(stream);
    break;
  default:
    break;
  }
}

/*******************   Heap and GC Stuff   **********************************/

static long heapSize = 0; // not thread-local
static const long heapMaxDelta = 1e6;
static const double heapSoftmaxGrowthThresh = 0.8; // grow softmax if >80% max after GC
static const double heapSoftmaxGrowthFactor = 1.4; // grow softmax by 40%

void THCSetGCHandler(THCState *state, void (*cutorchGCFunction_)(void *data), void *data )
{
  state->cutorchGCFunction = cutorchGCFunction_;
  state->cutorchGCData = data;
}

static long applyHeapDelta(THCState *state) {
  long newHeapSize = THAtomicAddLong(&heapSize, state->heapDelta) + state->heapDelta;
  state->heapDelta = 0;
  return newHeapSize;
}

// Here we maintain a dynamic softmax threshold for THC-allocated storages.
// When THC heap size goes above this softmax, the GC hook is triggered.
// If heap size is above 80% of the softmax after GC, then the softmax is
// increased.
static void maybeTriggerGC(THCState *state, long curHeapSize) {
  if (state->cutorchGCFunction != NULL && curHeapSize > state->heapSoftmax) {
    (state->cutorchGCFunction)(state->cutorchGCData);

    // ensure heapSize is accurate before updating heapSoftmax
    long newHeapSize = applyHeapDelta(state);

    if (newHeapSize > state->heapSoftmax * heapSoftmaxGrowthThresh) {
      state->heapSoftmax = state->heapSoftmax * heapSoftmaxGrowthFactor;
    }
  }
}

void THCHeapUpdate(THCState *state, long size) {
  state->heapDelta += size;
  // batch updates to global heapSize to minimize thread contention
  if (labs(state->heapDelta) < heapMaxDelta) {
    return;
  }

  long newHeapSize = applyHeapDelta(state);
  if (size > 0) {
    maybeTriggerGC(state, newHeapSize);
  }
}
