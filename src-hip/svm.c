#include "plat.h"

#include <dlfcn.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

/* ROCr only offers hipMemAddressReserve/hipMemMap when the KFD can manage compute
 * VM mappings through GEM_VA (KFD 1.15). On older kernels (e.g. SLES 6.4, KFD 1.14)
 * HIP managed memory gives the same fixed-address, partially resident range:
 * prefetching a range to the GPU backs it with VRAM, prefetching it back to the CPU
 * and discarding the host pages releases it.
 *
 * Without XNACK the GPU cannot fault, so KFD keeps every GPU accessible page backed
 * by RAM or VRAM. hipMallocManaged makes the whole range accessible to every GPU,
 * which would hold the unmapped part of each reservation in RAM. Reservations
 * therefore revoke GPU access through ROCr and grant it again per mapped range.
 *
 * HIP treats managed memory as host memory when choosing a copy path. Copies between
 * it and plain device memory are blits, but copies to or from the host (or between two
 * managed ranges) run on the CPU and fault the pages back into system RAM, where the
 * GPU then reads them over PCIe. The memcpy hooks route those through a device staging
 * buffer instead.
 */

#define KFD_TOPOLOGY "/sys/class/kfd/kfd/topology/nodes"
#define SVM_STAGING_SIZE (32 * M)

typedef hsa_status_t (*PFN_hsa_iterate_agents)(hsa_status_t (*callback)(hsa_agent_t, void *),
                                               void *data);
typedef hsa_status_t (*PFN_hsa_agent_get_info)(hsa_agent_t agent, int attribute, void *value);
typedef hsa_status_t (*PFN_hsa_amd_svm_attributes_set)(void *ptr, size_t size,
                                                       hsa_amd_svm_attribute_pair_t *attributes,
                                                       size_t attribute_count);

static PFN_hsa_iterate_agents p_hsa_iterate_agents;
static PFN_hsa_agent_get_info p_hsa_agent_get_info;
static PFN_hsa_amd_svm_attributes_set p_hsa_amd_svm_attributes_set;

#define SVM_MAX_GPU_AGENTS 16
static hsa_agent_t svm_gpu_agents[SVM_MAX_GPU_AGENTS];
static size_t svm_gpu_agent_count;

typedef struct SvmRange {
    CUdeviceptr va;
    size_t size;
    int device;
} SvmRange;

static pthread_mutex_t svm_lock = PTHREAD_MUTEX_INITIALIZER;
static SvmRange *svm_ranges;
static size_t svm_range_count;

static bool read_u64(const char *path, uint64_t *value) {
    FILE *f = fopen(path, "r");
    unsigned long long v;
    bool ok;

    if (!f) {
        return false;
    }
    ok = fscanf(f, "%llu", &v) == 1;
    fclose(f);
    *value = v;
    return ok;
}

static bool find_kfd_node(int domain, int location_id, unsigned *gpu_id, int *render_minor) {
    char path[128], key[64];
    unsigned long long value;

    for (int node = 0;; node++) {
        int node_domain = -1, node_location = -1, node_minor = -1;
        uint64_t node_gpu_id;
        FILE *f;

        snprintf(path, sizeof(path), KFD_TOPOLOGY "/%d/properties", node);
        if (!(f = fopen(path, "r"))) {
            return false;
        }
        while (fscanf(f, "%63s %llu", key, &value) == 2) {
            if (!strcmp(key, "domain")) {
                node_domain = (int)value;
            } else if (!strcmp(key, "location_id")) {
                node_location = (int)value;
            } else if (!strcmp(key, "drm_render_minor")) {
                node_minor = (int)value;
            }
        }
        fclose(f);

        snprintf(path, sizeof(path), KFD_TOPOLOGY "/%d/gpu_id", node);
        if (node_domain == domain && node_location == location_id &&
            read_u64(path, &node_gpu_id) && node_gpu_id) {
            *gpu_id = (unsigned)node_gpu_id;
            *render_minor = node_minor;
            return true;
        }
    }
}

static hsa_status_t collect_gpu_agent(hsa_agent_t agent, void *data) {
    int type;

    if (p_hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS &&
        type == HSA_DEVICE_TYPE_GPU && svm_gpu_agent_count < SVM_MAX_GPU_AGENTS) {
        svm_gpu_agents[svm_gpu_agent_count++] = agent;
    }
    return HSA_STATUS_SUCCESS;
}

static bool rocr_init(void) {
    static const char *const names[] = { "libhsa-runtime64.so.1" };
    void *module;

    if (p_hsa_amd_svm_attributes_set) {
        return true;
    }
    if (!(module = aimdo_find_loaded_module(names, ARRAY_SIZE(names))) ||
        !(p_hsa_iterate_agents = (PFN_hsa_iterate_agents)dlsym(module, "hsa_iterate_agents")) ||
        !(p_hsa_agent_get_info = (PFN_hsa_agent_get_info)dlsym(module, "hsa_agent_get_info")) ||
        !(p_hsa_amd_svm_attributes_set =
              (PFN_hsa_amd_svm_attributes_set)dlsym(module, "hsa_amd_svm_attributes_set"))) {
        log(AIMDO_LOG_ERROR, "%s: ROCr SVM entry points unavailable\n", __func__);
        p_hsa_amd_svm_attributes_set = NULL;
        return false;
    }
    if (p_hsa_iterate_agents(collect_gpu_agent, NULL) != HSA_STATUS_SUCCESS) {
        p_hsa_amd_svm_attributes_set = NULL;
        return false;
    }
    return true;
}

static bool find_gpu_agent(int domain, int location_id, uint64_t *handle) {
    for (size_t i = 0; i < svm_gpu_agent_count; i++) {
        uint32_t agent_domain, bdfid;

        if (p_hsa_agent_get_info(svm_gpu_agents[i], HSA_AMD_AGENT_INFO_DOMAIN, &agent_domain) ==
                HSA_STATUS_SUCCESS &&
            p_hsa_agent_get_info(svm_gpu_agents[i], HSA_AMD_AGENT_INFO_BDFID, &bdfid) ==
                HSA_STATUS_SUCCESS &&
            (int)agent_domain == domain && (int)bdfid == location_id) {
            *handle = svm_gpu_agents[i].handle;
            return true;
        }
    }
    return false;
}

static CUresult set_access(CUdeviceptr va, size_t size, uint64_t attribute) {
    hsa_amd_svm_attribute_pair_t pair = { attribute, svm_agent };

    if (p_hsa_amd_svm_attributes_set((void *)(uintptr_t)va, size, &pair, 1) != HSA_STATUS_SUCCESS) {
        log(AIMDO_LOG_ERROR, "%s: hsa_amd_svm_attributes_set(%#llx) failed\n", __func__,
            (ull)attribute);
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

/* The stream and staging buffer belong to dev, not to whatever device is current. */
static bool svm_create_device_objects(CUdevice dev) {
    CUdevice current;
    bool ok;

    if (!CHECK_CU(cuCtxGetDevice(&current)) || !CHECK_CU_ERROR(g_cuda.p_hipSetDevice(dev))) {
        return false;
    }
    /* hipExtMallocWithFlags is not hooked, so the staging buffer is accounted here. */
    ok = CHECK_CU_ERROR(cuStreamCreate(&svm_stream, CU_STREAM_NON_BLOCKING)) &&
         CHECK_CU_ERROR(cuEventCreate(&svm_staging_event, CU_EVENT_DISABLE_TIMING)) &&
         CHECK_CU_ERROR(g_cuda.p_hipExtMallocWithFlags(&svm_staging, SVM_STAGING_SIZE, 0));
    if (ok) {
        total_vram_usage += SVM_STAGING_SIZE;
    }
    CHECK_CU(g_cuda.p_hipSetDevice(current));
    return ok;
}

bool svm_init(CUdevice dev) {
    int vmm = 0, bus, device, domain, render_minor;
    unsigned gpu_id;

    if (!CHECK_CU(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev))) {
        return false;
    }
    if (vmm) {
        return true;
    }

    if (!CHECK_CU(cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev)) ||
        !CHECK_CU(cuDeviceGetAttribute(&device, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev)) ||
        !CHECK_CU(cuDeviceGetAttribute(&domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev))) {
        return false;
    }
    if (!rocr_init()) {
        return false;
    }
    if (!find_kfd_node(domain, bus << 8 | device << 3, &gpu_id, &render_minor) ||
        !find_gpu_agent(domain, bus << 8 | device << 3, &svm_agent)) {
        log(AIMDO_LOG_ERROR, "%s: no KFD node or ROCr agent for %04x:%02x:%02x\n", __func__,
            domain, bus, device);
        return false;
    }
    snprintf(svm_kfd_vram_path, sizeof(svm_kfd_vram_path), "/sys/class/kfd/kfd/proc/%d/vram_%u",
             (int)getpid(), gpu_id);
    snprintf(svm_drm_vram_path, sizeof(svm_drm_vram_path),
             "/sys/class/drm/renderD%d/device/mem_info_vram_used", render_minor);

    svm_backed = true;
    if (!svm_create_device_objects(dev)) {
        svm_cleanup();
        return false;
    }
    log(INFO, "comfy-aimdo: GPU has no HIP virtual memory support, backing VBARs with SVM\n");
    return true;
}

static void hip_free(CUdeviceptr va) {
    AimdoContext *devctx = g_devctx;

    /* hipFree is hooked, and the hook rebinds the device context. */
    CHECK_CU(g_cuda.p_cuMemFree_v2(va));
    set_devctx(devctx);
}

void svm_cleanup(void) {
    if (svm_staging) {
        hip_free(svm_staging);
        total_vram_usage -= SVM_STAGING_SIZE;
        svm_staging = 0;
    }
    if (svm_staging_event) {
        CHECK_CU(cuEventDestroy(svm_staging_event));
        svm_staging_event = NULL;
    }
    if (svm_stream) {
        CHECK_CU(cuStreamDestroy(svm_stream));
        svm_stream = NULL;
    }
    svm_backed = false;
}

CUresult svm_reserve(CUdeviceptr *va, size_t size) {
    hsa_amd_svm_attribute_pair_t revoke[SVM_MAX_GPU_AGENTS];
    SvmRange *ranges;
    CUresult err;

    if (!CHECK_CU_ERROR(err = cuMemAllocManaged(va, size, CU_MEM_ATTACH_GLOBAL))) {
        return err;
    }
    /* Fine grained SVM VRAM is mapped uncached. */
    if (!CHECK_CU_ERROR(err = cuMemAdvise(*va, size, HIP_MEM_ADVISE_SET_COARSE_GRAIN,
                                          g_devctx->_device_id))) {
        hip_free(*va);
        return err;
    }
    for (size_t i = 0; i < svm_gpu_agent_count; i++) {
        revoke[i] = (hsa_amd_svm_attribute_pair_t){ HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS,
                                                    svm_gpu_agents[i].handle };
    }
    if (p_hsa_amd_svm_attributes_set((void *)(uintptr_t)*va, size, revoke, svm_gpu_agent_count) !=
        HSA_STATUS_SUCCESS) {
        log(AIMDO_LOG_ERROR, "%s: could not revoke GPU access\n", __func__);
        hip_free(*va);
        return CUDA_ERROR_UNKNOWN;
    }
    /* hipMallocManaged populated the whole range; nothing needs it now. */
    madvise((void *)(uintptr_t)*va, size, MADV_DONTNEED);

    pthread_mutex_lock(&svm_lock);
    ranges = realloc(svm_ranges, (svm_range_count + 1) * sizeof(*svm_ranges));
    if (ranges) {
        svm_ranges = ranges;
        svm_ranges[svm_range_count++] = (SvmRange){ *va, size, g_devctx->_device_id };
    }
    pthread_mutex_unlock(&svm_lock);

    if (!ranges) {
        log(CRITICAL, "Host OOM\n");
        hip_free(*va);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}

void svm_free(CUdeviceptr va) {
    pthread_mutex_lock(&svm_lock);
    for (size_t i = 0; i < svm_range_count; i++) {
        if (svm_ranges[i].va == va) {
            svm_ranges[i] = svm_ranges[--svm_range_count];
            break;
        }
    }
    pthread_mutex_unlock(&svm_lock);
    hip_free(va);
}

CUresult svm_map(CUdeviceptr va, size_t size, int device) {
    uint64_t before, after;

    if (!read_u64(svm_kfd_vram_path, &before)) {
        log(AIMDO_LOG_ERROR, "%s: cannot read %s\n", __func__, svm_kfd_vram_path);
        return CUDA_ERROR_UNKNOWN;
    }
    if (set_access(va, size, HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE) != CUDA_SUCCESS ||
        !CHECK_CU_ERROR(cuMemPrefetchAsync(va, size, device, svm_stream)) ||
        !CHECK_CU_ERROR(cuStreamSynchronize(svm_stream))) {
        return CUDA_ERROR_UNKNOWN;
    }
    /* KFD reports success when VRAM runs out and leaves the rest in system RAM. */
    if (read_u64(svm_kfd_vram_path, &after) && (int64_t)(after - before) < (int64_t)size) {
        log(DEBUG, "%s: only %zuk of %zuk reached VRAM\n", __func__,
            (size_t)(after - before) / K, size / K);
        svm_unmap(va, size);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}

void svm_unmap(CUdeviceptr va, size_t size) {
    if (!size) {
        return;
    }
    /* Only prefetching out frees the VRAM. Without GPU access the host copy can be dropped. */
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU_ERROR(cuMemPrefetchAsync(va, size, CU_DEVICE_CPU, svm_stream));
    CHECK_CU_ERROR(cuStreamSynchronize(svm_stream));
    set_access(va, size, HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS);
    madvise((void *)(uintptr_t)va, size, MADV_DONTNEED);
}

/* hipMemGetInfo does not count SVM pages resident in VRAM. */
bool svm_mem_get_info(size_t *free_bytes, size_t *total_bytes) {
    uint64_t used;

    if (!read_u64(svm_drm_vram_path, &used)) {
        return false;
    }
    *total_bytes = vram_capacity;
    *free_bytes = used < vram_capacity ? vram_capacity - used : 0;
    return true;
}

/* Caller holds svm_lock. */
static int svm_range_device(const void *ptr, size_t size) {
    CUdeviceptr p = (CUdeviceptr)(uintptr_t)ptr;

    for (size_t i = 0; i < svm_range_count; i++) {
        if (p < svm_ranges[i].va + svm_ranges[i].size && svm_ranges[i].va < p + size) {
            return svm_ranges[i].device;
        }
    }
    return -1;
}

CUresult svm_memcpy(void *dst, const void *src, size_t size, int kind, CUstream stream,
                    PFN_hipMemcpyAsync copy) {
    AimdoContext *devctx = g_devctx;
    int dst_device, src_device;
    CUresult err = CUDA_SUCCESS;

    if (!svm_range_count || kind == HIP_MEMCPY_HOST_TO_HOST) {
        return copy(dst, src, size, kind, stream);
    }

    pthread_mutex_lock(&svm_lock);
    dst_device = svm_range_device(dst, size);
    src_device = svm_range_device(src, size);
    if ((dst_device < 0 && src_device < 0) ||
        (kind == HIP_MEMCPY_DEVICE_TO_DEVICE && (dst_device < 0 || src_device < 0))) {
        pthread_mutex_unlock(&svm_lock);
        return copy(dst, src, size, kind, stream);
    }

    set_devctx_for_device(dst_device >= 0 ? dst_device : src_device);
    for (size_t offset = 0; offset < size; offset += SVM_STAGING_SIZE) {
        size_t n = MIN(SVM_STAGING_SIZE, size - offset);
        void *staging = (void *)(uintptr_t)svm_staging;

        /* The staging buffer is shared by every stream; wait for its last user. */
        if (!CHECK_CU_ERROR(err = cuStreamWaitEvent(stream, svm_staging_event, 0)) ||
            !CHECK_CU_ERROR(err = copy(staging, (const char *)src + offset, n,
                                       src_device >= 0 ? HIP_MEMCPY_DEVICE_TO_DEVICE : kind, stream)) ||
            !CHECK_CU_ERROR(err = copy((char *)dst + offset, staging, n,
                                       dst_device >= 0 ? HIP_MEMCPY_DEVICE_TO_DEVICE : kind, stream)) ||
            !CHECK_CU_ERROR(err = cuEventRecord(svm_staging_event, stream))) {
            break;
        }
    }
    set_devctx(devctx);
    pthread_mutex_unlock(&svm_lock);
    return err;
}
