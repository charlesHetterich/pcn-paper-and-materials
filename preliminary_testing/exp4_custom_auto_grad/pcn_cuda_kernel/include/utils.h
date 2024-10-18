#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor pcn_fw_cu(
    torch::Tensor c,
    torch::Tensor l,
    torch::Tensor lnext,
    torch::Tensor b);

std::vector<torch::Tensor> pcn_bw_cu(
    const torch::Tensor grad_cnext,
    const torch::Tensor c,
    const torch::Tensor l,
    const torch::Tensor lnext,
    const torch::Tensor bias);