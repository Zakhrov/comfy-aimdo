#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__HIP_PLATFORM_AMD__)
/* HIP runtime exports use the platform C ABI. */
#define CUDAAPI
#elif defined(_WIN32) || defined(_WIN64)
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

typedef unsigned long long cuuint64_t;

#if defined(_WIN64) || defined(__LP64__) || defined(_LP64)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif

typedef int CUdevice;
typedef int CUresult;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef unsigned long long CUmemGenericAllocationHandle;

typedef struct CUuuid_st {
    char bytes[16];
} CUuuid;

/* hipDeviceAttribute_t is not numerically compatible with CUdevice_attribute.
 * hipDeviceAttributeIntegrated is 16, so querying a HIP device with the CUDA
 * value would read hipDeviceAttributeKernelExecTimeout instead.
 */
typedef enum CUdevice_attribute_enum {
#if defined(__HIP_PLATFORM_AMD__)
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 16,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 67,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 68,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 69,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 89,
#else
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
#endif
} CUdevice_attribute;

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_UNKNOWN = 999,
} cudaError_enum;

#define CU_DEVICE_CPU ((CUdevice)-1)

typedef enum CUstream_flags_enum {
    CU_STREAM_NON_BLOCKING = 0x1,
} CUstream_flags;

typedef enum CUmemAttach_flags_enum {
    CU_MEM_ATTACH_GLOBAL = 0x1,
} CUmemAttach_flags;

#if defined(__HIP_PLATFORM_AMD__)
/* HIP only: lets the device cache managed memory resident in its VRAM. */
#define HIP_MEM_ADVISE_SET_COARSE_GRAIN 100

typedef enum hipMemcpyKind_enum {
    HIP_MEMCPY_HOST_TO_HOST = 0,
    HIP_MEMCPY_HOST_TO_DEVICE = 1,
    HIP_MEMCPY_DEVICE_TO_HOST = 2,
    HIP_MEMCPY_DEVICE_TO_DEVICE = 3,
} hipMemcpyKind;

/* ROCr, for per-agent SVM access. */
typedef int hsa_status_t;
typedef struct hsa_agent_s {
    uint64_t handle;
} hsa_agent_t;
typedef struct hsa_amd_svm_attribute_pair_s {
    uint64_t attribute;
    uint64_t value;
} hsa_amd_svm_attribute_pair_t;

#define HSA_STATUS_SUCCESS 0
#define HSA_AGENT_INFO_DEVICE 17
#define HSA_DEVICE_TYPE_GPU 1
#define HSA_AMD_AGENT_INFO_BDFID 0xA006
#define HSA_AMD_AGENT_INFO_DOMAIN 0xA00F
#define HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE 0x200
#define HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS 0x202
#endif

typedef enum CUevent_flags_enum {
    CU_EVENT_DISABLE_TIMING = 0x2,
} CUevent_flags;

typedef enum CUmemAllocationHandleType_enum {
    CU_MEM_HANDLE_TYPE_NONE = 0x0,
} CUmemAllocationHandleType;

typedef enum CUmemAccess_flags_enum {
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0x0,
    CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3,
} CUmemAccess_flags;

typedef enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0x0,
    CU_MEM_LOCATION_TYPE_DEVICE = 0x1,
    CU_MEM_LOCATION_TYPE_HOST = 0x2,
    CU_MEM_LOCATION_TYPE_HOST_NUMA = 0x3,
    CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 0x4,
} CUmemLocationType;

typedef enum CUmemAllocationType_enum {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,
    CU_MEM_ALLOCATION_TYPE_PINNED = 0x1,
} CUmemAllocationType;

typedef struct CUmemLocation_st {
    CUmemLocationType type;
    int id;
} CUmemLocation;

typedef struct CUmemAllocationProp_st {
    CUmemAllocationType type;
    CUmemAllocationHandleType requestedHandleTypes;
    CUmemLocation location;
    void *win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
} CUmemAllocationProp;

typedef struct CUmemAccessDesc_st {
    CUmemLocation location;
    CUmemAccess_flags flags;
} CUmemAccessDesc;

typedef enum CUdriverProcAddress_flags_enum {
    CU_GET_PROC_ADDRESS_DEFAULT = 0x0,
    CU_GET_PROC_ADDRESS_LEGACY_STREAM = 0x1,
    CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = 0x2,
} CUdriverProcAddress_flags;

typedef enum CUdriverProcAddressQueryResult_enum {
    CU_GET_PROC_ADDRESS_SUCCESS = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2,
} CUdriverProcAddressQueryResult;
