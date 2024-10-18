#include <torch/extension.h>
#include "utils.h"

torch::Tensor pcn_forward(
    torch::Tensor c,
    torch::Tensor l,
    torch::Tensor lnext,
    torch::Tensor b)
{
    // Check input shapes
    CHECK_INPUT(c);
    CHECK_INPUT(l);
    CHECK_INPUT(lnext);
    CHECK_INPUT(b);

    TORCH_CHECK(c.ndimension() == 2, "c should be 2D (batch_size, in_neurons)");
    TORCH_CHECK(l.ndimension() == 2, "l should be 2D (in_neurons, dimensions)");
    TORCH_CHECK(lnext.ndimension() == 2, "lnext should be 2D (out_neurons, dimensions)");
    TORCH_CHECK(b.ndimension() == 1, "b should be 1D (out_neurons)");

    TORCH_CHECK(c.size(1) == l.size(0), "wrong input_features size");
    TORCH_CHECK(l.size(1) == lnext.size(1), "layer dimensions should be the same");
    TORCH_CHECK(lnext.size(0) == b.size(0), "lnext and bias, b, should have the same number of features");

    return pcn_fw_cu(c, l, lnext, b);
}

std::vector<torch::Tensor> pcn_backward(
    const torch::Tensor grad_cnext,
    const torch::Tensor c,
    const torch::Tensor l,
    const torch::Tensor lnext,
    const torch::Tensor bias)
{
    // Check input shapes
    CHECK_INPUT(grad_cnext);
    CHECK_INPUT(c);
    CHECK_INPUT(l);
    CHECK_INPUT(lnext);
    CHECK_INPUT(bias);

    TORCH_CHECK(grad_cnext.ndimension() == 2, "grad_cnext should be 2D (batch_size, out_neurons)");
    TORCH_CHECK(c.ndimension() == 2, "c should be 2D (batch_size, in_neurons)");
    TORCH_CHECK(l.ndimension() == 2, "l should be 2D (in_neurons, dimensions)");
    TORCH_CHECK(lnext.ndimension() == 2, "lnext should be 2D (out_neurons, dimensions)");
    TORCH_CHECK(bias.ndimension() == 1, "bias should be 1D (out_neurons)");

    TORCH_CHECK(grad_cnext.size(0) == c.size(0), "batch_size should be the same");
    TORCH_CHECK(grad_cnext.size(1) == lnext.size(0), "wrong output_features size");
    TORCH_CHECK(c.size(1) == l.size(0), "wrong input_features size");
    TORCH_CHECK(l.size(1) == lnext.size(1), "layer dimensions should be the same");
    TORCH_CHECK(lnext.size(0) == bias.size(0), "lnext and bias, b, should have the same number of features");

    return pcn_bw_cu(grad_cnext, c, l, lnext, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &pcn_forward, "PCN forward");
    m.def("backward", &pcn_backward, "PCN backward");
}