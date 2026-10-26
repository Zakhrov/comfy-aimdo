/* Shared hook declarations for the CUDA hook backends.
 *
 * This is intentionally an includable implementation header, not a normal
 * interface header. It defines static functions and static data, and should
 * only be included directly by one backend .c file per translation unit.
 */

#pragma once

typedef struct {
    void **true_ptr;
    void **target_ptr;
    void *hook_ptr;
    const char *name;
} HookEntry;

static CUresult (CUDAAPI *true_cuMemAlloc_v2)(CUdeviceptr*, size_t);
static CUresult (CUDAAPI *true_cuMemFree_v2)(CUdeviceptr);
static CUresult (CUDAAPI *true_cuMemAllocAsync)(CUdeviceptr*, size_t, CUstream);
static CUresult (CUDAAPI *true_cuMemAllocAsync_ptsz)(CUdeviceptr*, size_t, CUstream);
static CUresult (CUDAAPI *true_cuMemFreeAsync)(CUdeviceptr, CUstream);
static CUresult (CUDAAPI *true_cuMemFreeAsync_ptsz)(CUdeviceptr, CUstream);

static CUresult CUDAAPI aimdo_cuMemAlloc_v2(CUdeviceptr *dptr, size_t size) {
    return aimdo_cuda_malloc(dptr, size, true_cuMemAlloc_v2);
}

static CUresult CUDAAPI aimdo_cuMemFree_v2(CUdeviceptr dptr) {
    return aimdo_cuda_free(dptr, true_cuMemFree_v2);
}

static CUresult CUDAAPI aimdo_cuMemAllocAsync(CUdeviceptr *dptr, size_t size, CUstream hStream) {
    return aimdo_cuda_malloc_async(dptr, size, hStream, true_cuMemAllocAsync);
}

static CUresult CUDAAPI aimdo_cuMemAllocAsync_ptsz(CUdeviceptr *dptr, size_t size, CUstream hStream) {
    return aimdo_cuda_malloc_async(dptr, size, hStream, true_cuMemAllocAsync_ptsz);
}

static CUresult CUDAAPI aimdo_cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    return aimdo_cuda_free_async(dptr, hStream, true_cuMemFreeAsync);
}

static CUresult CUDAAPI aimdo_cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream) {
    return aimdo_cuda_free_async(dptr, hStream, true_cuMemFreeAsync_ptsz);
}

#if defined(__HIP_PLATFORM_AMD__) && !defined(_WIN32) && !defined(_WIN64)
static CUresult (CUDAAPI *true_hipMemcpyAsync)(void *, const void *, size_t, int, CUstream);
static CUresult (CUDAAPI *true_hipMemcpyWithStream)(void *, const void *, size_t, int, CUstream);
static CUresult (CUDAAPI *true_cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
static CUresult (CUDAAPI *true_cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
static CUresult (CUDAAPI *true_cuMemGetInfo)(size_t *, size_t *);

static CUresult CUDAAPI aimdo_hipMemcpyAsync(void *dst, const void *src, size_t bytes, int kind,
                                             CUstream hStream) {
    return svm_memcpy(dst, src, bytes, kind, hStream, true_hipMemcpyAsync);
}

static CUresult CUDAAPI aimdo_hipMemcpyWithStream(void *dst, const void *src, size_t bytes,
                                                  int kind, CUstream hStream) {
    return svm_memcpy(dst, src, bytes, kind, hStream, true_hipMemcpyWithStream);
}

static CUresult CUDAAPI aimdo_cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, size_t bytes,
                                                CUstream hStream) {
    return svm_memcpy((void *)(uintptr_t)dst, src, bytes, HIP_MEMCPY_HOST_TO_DEVICE, hStream,
                      true_hipMemcpyAsync);
}

/* hipMemGetInfo leaves out SVM pages resident in VRAM; report what the device has. */
static CUresult CUDAAPI aimdo_cuMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
    AimdoContext *devctx = g_devctx;
    bool answered = set_devctx_for_current_cuda_device() && svm_backed &&
                    svm_mem_get_info(free_bytes, total_bytes);

    set_devctx(devctx);
    return answered ? CUDA_SUCCESS : true_cuMemGetInfo(free_bytes, total_bytes);
}

static CUresult CUDAAPI aimdo_cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, size_t bytes,
                                                CUstream hStream) {
    return svm_memcpy(dst, (const void *)(uintptr_t)src, bytes, HIP_MEMCPY_DEVICE_TO_HOST, hStream,
                      true_hipMemcpyAsync);
}
#endif

static const HookEntry hooks[] = {
    { (void **)&true_cuMemAlloc_v2,        (void **)&g_cuda.p_cuMemAlloc_v2,        aimdo_cuMemAlloc_v2,        "cuMemAlloc_v2" },
    { (void **)&true_cuMemFree_v2,         (void **)&g_cuda.p_cuMemFree_v2,         aimdo_cuMemFree_v2,         "cuMemFree_v2" },
    { (void **)&true_cuMemAllocAsync,      (void **)&g_cuda.p_cuMemAllocAsync,      aimdo_cuMemAllocAsync,      "cuMemAllocAsync" },
    { (void **)&true_cuMemAllocAsync_ptsz, (void **)&g_cuda.p_cuMemAllocAsync_ptsz, aimdo_cuMemAllocAsync_ptsz, "cuMemAllocAsync_ptsz" },
    { (void **)&true_cuMemFreeAsync,       (void **)&g_cuda.p_cuMemFreeAsync,       aimdo_cuMemFreeAsync,       "cuMemFreeAsync" },
    { (void **)&true_cuMemFreeAsync_ptsz,  (void **)&g_cuda.p_cuMemFreeAsync_ptsz,  aimdo_cuMemFreeAsync_ptsz,  "cuMemFreeAsync_ptsz" },
#if defined(__HIP_PLATFORM_AMD__) && !defined(_WIN32) && !defined(_WIN64)
    { (void **)&true_hipMemcpyAsync,       (void **)&g_cuda.p_hipMemcpyAsync,       aimdo_hipMemcpyAsync,       "hipMemcpyAsync" },
    { (void **)&true_hipMemcpyWithStream,  (void **)&g_cuda.p_hipMemcpyWithStream,  aimdo_hipMemcpyWithStream,  "hipMemcpyWithStream" },
    { (void **)&true_cuMemcpyHtoDAsync,    (void **)&g_cuda.p_cuMemcpyHtoDAsync,    aimdo_cuMemcpyHtoDAsync,    "cuMemcpyHtoDAsync" },
    { (void **)&true_cuMemcpyDtoHAsync,    (void **)&g_cuda.p_cuMemcpyDtoHAsync,    aimdo_cuMemcpyDtoHAsync,    "cuMemcpyDtoHAsync" },
    { (void **)&true_cuMemGetInfo,         (void **)&g_cuda.p_cuMemGetInfo,         aimdo_cuMemGetInfo,         "cuMemGetInfo" },
#endif
};
