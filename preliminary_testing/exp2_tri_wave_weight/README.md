# Experiment 2: Triangle Wave Distance-Weight Generation Function

The idea of this experiment is to pass our raw distance matrix through a triangle wave in order to generate our final weight.

In prior exeriments the function of distance that we used was to raise the raw distance by a power (`-1/2`, `-1`, `-2`, `-4`, etc). The issue here is that our weights are absolute values, which greatly limits the complexity of information transfer possible, which in turn limits the depth of optimality able to be achieved by this network.

It is known that in the brain, a synapse can be excitory, passing positive charge to a neuron, or inhibitory, passing negative charge to a neuron (**spiking nets paper**). We also observe both positive and negative weights in virtually all modern ANN's, allowing the passing of negative values, as well as tha ability to negate passed values. Thus, we should consider this capability to pass negative information as a necessetity to achieve complex knowledge transfer.

We hope that by using a tiangle wave as our distance-weight function, this will enable necessary negation-of-charge capability, without having to give up any of the weight-generation power of the distance matrix calculation. A triangle wav also has the advantage of a linear form, which hopefully makes it easy to differentiate through, as opposed to other wave forms. Sin waves have flat points and inconsistent derivatives. Saw and Square waves are disconnected, which might make the network make un-intended jumps in weight. Triangle waves have a derivative of either `-1` or `1`. At non-differetiable points, we can arbitrarily assign a subgradient of `-1` or `1`.

- the initial plan is to use a triangle wave centered around `0`, on the range `[-1, 1]`
- We will have to play around with the frequency of the triangle wave. I imagine that this should be informed by the density of the network (# of neurons vs. space we generate within)
