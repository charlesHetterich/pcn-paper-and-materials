# Experiment 4: Custom Auto Grad

The goal of this test notebook is to develop a custom `nn.Module` that can be used to represent a pcn layer. The intention would be to build out custom auto-differentiation for this layer s.t. we never have to store the full distance matrix of any layer in memory at once, during training or eval. The plan would be:

- Load layer position + bias params into gpu memory.
- Load input data into memory
- currently at a given layer, $l$, of the network, in a forward pass:
  - currently, a full distance matrix is computed, passed through a tri-wave, and then multiplied by input.
  - the full distance matrix, during training, is stored in memory for each layer. This ends up with just as much, if not slightly more, memory being used to store the model in comparison to a regular FCN.
  - One major advantages of PCN's that we can, in theory, take up a very small amount of space on a GPU with the network, in comparison to a regular FCN. So it is important to look into ways to actually minimize the memory taken up by the network.
- What we can do in a custom `nn.Module` to minimize the network size:
  - calculate the distance between all points in the current layer, $P_l$, and a single point in the next layer, $P_{l+1,i} \forall{i}$. Use this information to calculate the value, or **current**, for the neuron on the next layer $c_{l+1,i}$. Perhaps such as
  - $c_{l+1,i} = \sigma(F(D_i)c_i^\top + b_i)$
  - Do these operations in parallel, on the GPU
  - wait to collect all outputs until moving onto the next layer
  - memory used should just be taken by the # of parallel instances of this operation that are being performed at once, where each one of those instances holds $N_l$ new values where $N_l$ is the number of neurons in the current layer.
  - When an instance of this operation is done, we should replace the distances stored in this operation with distances calculated from a new instance of this operation.
  - This should keep a pretty tight throttle on the amount of memory taken up in a forward pass
  - When training and including a backward pass, the amount of storage should not change.
  - for the backward step, instead of storing all of the distances calculated in the forward step to use during auto-diff, re-calculated thos distances on the fly during the backward step, as needed.
  - Similarly to the forward step, have parellel instances of the backward operation happening where each again has a set of distances to calculate and use, but throw away after that instances of the backward operation is completed.
