# Awesome LLM Papers [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 

# Papers

Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C. C., Xu, M., ... & Li, S. [**Pytorch fsdp: experiences on scaling fully sharded data parallel**](https://arxiv.org/pdf/2304.11277) arXiv 2023
<details>
  <summary>Notes</summary>

  - Presents a new framework for training large (>30B) parameter models across several GPUs
  - Pipeline Parallelism (Vertical Split): Partitions a model instance into stages, with each stage being processed on one instance. For example first layer being processed by first GPU, second layer by second GPU etc...
  - Tensor parallelism (Horizontal Split): Splits up the model parameters (e.g. within one layer, thats why its horizontal) and processes these across several GPUs. Communicates at layer boundaries (e.g. going from layer 1 to layer 2)
  - Zero-redundancy Parallelism: Also splits up the parameters but communicates parameters on-demand. So I think the difference is that Zero communicates parameters, while TP communicates activations. 
  - Their technique called FSDP is very similar to Zero but adapted to the Pytorch framework. ![FSDP Overview](./assets/fsdp.png) 
  - FSDP communicates parameters before computation, while TP communicates activations after partial computation.
  - There are different sharding strategies, which they parametrize using the sharding factor F:
    - if F is 1, the model is fully replicated on each device
    - if F is W (the number of available GPUs aka world size) then the model is fully sharded. This has the lowest memory footprint with the highest communication overhead.
    - If F is between 1 and W, they call it hybrid sharding
  - Communication Strategies:
    - Overlapping:
      - GPU can fetch the parameters for the next layer while still computing the current one, so it runs on a different process
    - Backward Pre-fetching: 
      - To avoid a communication bottleneck during the backward pass, they issue the request for the next layer's parameters before it has finished calculating the current layer.
    - Forward Pre-fetching: 
      - When the model is static, the execution order from the previous training iteration can be used to fetch (another layers) parameters even before the forward pass for that layer begins.
</details>