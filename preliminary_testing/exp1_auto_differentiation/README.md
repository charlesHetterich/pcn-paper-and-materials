# Experiment 1: Auto Differentiation

The core benefit of a PCN is to generate weights from the positional information of neurons. Feed forward-fixed PCN's can be represented exactly as a regular modern deep neural network. This alone shows that we can use backpropagation to learn new neuron positions. We hope to show in this first exeriment that we can do this by using tensorflow's already built auto-differentiation feature.

In order for a PCN to be competitive with modern DNN's, it must be compatible with cuda vector acceleration. This should be completely possible if we store each layer as a tensor of neuron positions. We can then use `torch.cdist` to generate the distance matrix between the current layer and the next.

In theory, we should be able to complete a full cycle of a `forward pass -> backprop pass`, while only ever having a single distance matrix loaded at a time. Depending on the internal workings of tensorflows computation graph, this might require some workarounds. But the fact that the distance matrix is given by the positional information proves that we can always lazy-load a distance matrix only when needed in both forward and backward passes.

This will be a major advantage of PCN's over current DNN's, considering that the available memory on a GPU to store parameters is one of the current biggest bottlenecks in deep learning.

In this experiment, we will first look at a small PCN and calcualte it's position updates both manually as well as using auto-differentiation, to confirm that auto-differentiation is calculating updates correctly. We will then look at some larger scale examples to experiment with the effect on GPU storage both during `.eval()` and `.train()` modes.
