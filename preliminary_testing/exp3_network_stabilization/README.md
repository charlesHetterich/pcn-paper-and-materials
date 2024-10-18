# Experiment 3: Network Stabilization

Like we eluded to at the end of the previous experiment, the next step is to stabilize our network so that it is resiliant to changes in several of our hyper-parameters.

Specifically we are aiming to to be able to scale out the size/depth of our network. We should monitor that each layer is learning properly given various different sizes/shapes. We also need to focus on our learn rate, triangle wave amplitude, and triangle wave period. We hope to solve some maths to make our choice of these parameters to be rigorous rather than chosen arbitrarily via intuition.
