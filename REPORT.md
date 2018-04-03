# Semantic Segmentation Project

---

Implementation of Semantic Segmentation using fully convolutional network as specified in [this](https://arxiv.org/pdf/1411.4038.pdf) paper.

Network architecture:

![arch](./report_images/arch.png)


Key points:

- Used `truncated_normal_initializer` kernel initializers to initialize the weights.
- Used `l2_regularizer` for regularization.
- Use `AdamOptimizer`
- Tried different batch sizes and epochs. Used [this](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network) article to decide on best parameters

Loss Curve:

![loss](report_images/loss.png)

Example Segmentation Results:

![um_000006.png](report_images/um_000006.png)
![um_000013.png](report_images/um_000013.png)
![um_000014.png](report_images/um_000014.png)
![um_000018.png](report_images/um_000018.png)
![um_000029.png](report_images/um_000029.png)
![um_000032.png](report_images/um_000032.png)
![um_000039.png](report_images/um_000039.png)
