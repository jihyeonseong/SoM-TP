# Towards Diverse Perspective Learning with Selection over Multiple Temporal Poolings (AAAI24)
* This is the author code implements "Towards Diverse Perspective Learning with Selection over Multiple Temporal Poolings," a paper accepted at AAAI 2024.
* It builds upon the official code of [DTP github](https://github.com/donalee/DTW-Pool) and [softDTW github](https://github.com/Maghoumi/pytorch-softdtw-cuda) based on PyTorch.
* For further details, please refer to the original [DTP](https://arxiv.org/abs/2104.02577) and [softDTW](https://arxiv.org/abs/1703.01541) papers.
## Overview
![image](https://github.com/jihyeonseong/SoM-TP/assets/159874470/c15390f3-3e6c-477b-b019-3ae1b08bda3f)
In Time Series Classification (TSC), temporal pooling methods that consider sequential information have been proposed. However, we found that each temporal pooling has a distinct mechanism, and can perform better or worse depending on time series data. We term this fixed pooling mechanism a single perspective of temporal poolings. In this paper, we propose a novel temporal pooling method with diverse perspective learning: Selection over Multiple Temporal Poolings (SoM-TP). 
* We investigate data dependency arising from distinct perspectives of existing temporal poolings.
* We propose SoM-TP, a new temporal pooling method that fully utilizes the diverse temporal pooling mechanisms through an MCL-inspired selection ensemble.
* We employ an attention mechanism to enable a non-iterative ensemble in a single classifier.
* We define DPLN and perspective loss as a regularizer to promote diverse pooling selection.
