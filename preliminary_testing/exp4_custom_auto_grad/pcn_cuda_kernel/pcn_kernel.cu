#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// If we define our neuron positions randomly on the range [-1, 1], a period of 0.1
// seems sufficiently small in order for the distances to cover a full period, regardless
// of layer size & point dimensionality.
#define FREQ 10.0

// We need this function so that we can do the big three
// - Shared memory
// - Dynamic Allocation
// - Generic Type
template <typename scalar_t>
__device__ scalar_t *shared_memory_proxy()
{
    // do we need an __align__() here? I don't think so...
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<scalar_t *>(memory);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t tri_wave(scalar_t z)
{
    return 1.0 - 2.0 * abs(fmodf(z, 2.0f) - 1.0);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tri_wave(scalar_t z)
{
    return fmodf(z, 2.0f) < 1 ? 2 : -2;
}

template <typename scalar_t>
__global__ void pcn_fw_kernel(
    const int N,
    const int B,
    const int D,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> c,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> l,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> lnext,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> bias,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> cnext)
{
    // Setup shared memory, get indices, validate indices
    auto agg_result = shared_memory_proxy<scalar_t>();
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (f >= lnext.size(0) || n >= l.size(0))
        return;

    // Initialize shared  once per block
    if (threadIdx.y == 0)
    {
        for (int b = 0; b < B; ++b)
        {
            const int s_idx = b * blockDim.x + threadIdx.x;
            agg_result[s_idx] = 0.0;
            // Initialize cnext if we are on first block
            if (blockIdx.y == 0)
            {
                cnext[b][f] = bias[f];
            }
        }
    }
    __syncthreads();

    // Rest of kernel
    for (int b = 0; b < B; ++b)
    {
        const int s_idx = b * blockDim.x + threadIdx.x;
        scalar_t sqr_dst = 0.0;
        for (int d = 0; d < D; ++d)
        {
            scalar_t diff = l[n][d] - lnext[f][d];
            sqr_dst += diff * diff;
        }
        atomicAdd(&agg_result[s_idx], c[b][n] * (tri_wave(FREQ * sqrt(sqr_dst)) / sqrt((scalar_t)N)));
    }
    __syncthreads();

    // Add agg_result to cnext once per block
    if (threadIdx.y == 0)
    {
        for (int b = 0; b < B; ++b)
        {
            const int s_idx = b * blockDim.x + threadIdx.x;
            atomicAdd(&cnext[b][f], agg_result[s_idx]);
        }
    }
}

torch::Tensor pcn_fw_cu(
    const torch::Tensor c,
    const torch::Tensor l,
    const torch::Tensor lnext,
    const torch::Tensor bias)
{
    const int B = c.size(0), F = lnext.size(0), N = l.size(0), D = l.size(1);

    torch::Tensor cnext = torch::zeros({B, F}, c.options());

    const int DIM_X = 4;
    const int DIM_Y = 256;

    const dim3 threads(DIM_X, DIM_Y);
    const dim3 blocks((F + threads.x - 1) / threads.x,
                      (N + threads.y - 1) / threads.y);

    size_t SHMEM = B * DIM_X * sizeof(float); // #* sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(c.type(), "pcn_fw_cu", ([&]
                                                       { pcn_fw_kernel<scalar_t><<<blocks, threads, SHMEM>>>(
                                                             N,
                                                             B,
                                                             D,
                                                             c.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                             l.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                             lnext.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                             bias.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                             cnext.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));

    return cnext;
}

template <typename scalar_t>
__global__ void pcn_bw_kernel(
    const int N,
    const int B,
    const int D,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_cnext,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> c,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> l,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> lnext,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> bias,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_c,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_l,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_lnext,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> grad_bias)
{

    // TODO : Add shared memory to do most addition operations in

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= l.size(0) || f >= lnext.size(0))
        return; // validation

    for (int b = 0; b < B; ++b)
    {
        if (n == 0)
        {
            atomicAdd(&grad_bias[f], grad_cnext[b][f]);
        }

        scalar_t dst = 0.0;
        for (int d = 0; d < D; ++d)
        {
            scalar_t diff = l[n][d] - lnext[f][d];
            dst += diff * diff;
        }

        for (int d = 0; d < D; ++d)
        {
            scalar_t d_dst = (l[n][d] - lnext[f][d]) / sqrt(dst);
            scalar_t tmp = grad_cnext[b][f] * c[b][n] * (1 / sqrt((scalar_t)N)) * FREQ * d_tri_wave(FREQ * sqrt(dst));

            atomicAdd(&grad_l[n][d], tmp * d_dst);
            atomicAdd(&grad_lnext[f][d], -tmp * d_dst);
        }

        atomicAdd(&grad_c[b][n], grad_cnext[b][f] * (tri_wave(FREQ * sqrt(dst)) / sqrt((scalar_t)N)));
    }
}

std::vector<torch::Tensor> pcn_bw_cu(
    const torch::Tensor grad_cnext,
    const torch::Tensor c,
    const torch::Tensor l,
    const torch::Tensor lnext,
    const torch::Tensor bias)
{
    const int B = c.size(0), F = lnext.size(0), N = l.size(0), D = l.size(1);

    torch::Tensor grad_c = torch::zeros({B, N}, l.options());
    torch::Tensor grad_l = torch::zeros({N, D}, l.options());
    torch::Tensor grad_lnext = torch::zeros({F, D}, lnext.options());
    torch::Tensor grad_bias = torch::zeros({F}, bias.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x,
                      (F + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(c.type(), "pcn_bw_cu", ([&]
                                                       { pcn_bw_kernel<scalar_t><<<blocks, threads>>>(
                                                             N,
                                                             B,
                                                             D,
                                                             grad_cnext.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                             c.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                             l.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                             lnext.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                             bias.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                             grad_c.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                             grad_l.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                             grad_lnext.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                             grad_bias.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()); }));
    return {grad_c, grad_l, grad_lnext, grad_bias};
}