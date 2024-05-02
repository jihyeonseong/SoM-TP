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
## Running the codes
### STEP 1. Download the benchmark datsets for time series classification
* The datasets can be downloaded form the [UCR/UEA repository](https://www.timeseriesclassification.com/).
* Create a directory named "data" and store downloaded datasets within it.
### STEP 2. Train the CNN classifier with various temporal poolings including SoM-TP
For traditional temporal poolings,
```
python main.py --model=ConvPool --pool=DTP
```
and for SoM-TP
```
python main.py --model=SoMTP
```
### STEP 3. Run LRP (Layer-wise Relevance Propagation: XAI input attribution method)
For traditional temporal poolings,
```
python LRP.py --model=ConvPool --pool=DTP
```
and for SoM-TP
```
python LRP.py --model=SoMTP
```
### SoM-TP performance
1. Comparison with traditional temporal poolings
![image](https://github.com/jihyeonseong/SoM-TP/assets/159874470/c9b862b8-7b2f-45eb-ad12-ca12af0ac7e0)
2. Comparison with advanced TSC methods
![image](https://github.com/jihyeonseong/SoM-TP/assets/159874470/dd0eb53a-f287-4943-bd62-85be70ac65c6)
3. SoM-TP dynamic selection 
![image](https://github.com/jihyeonseong/SoM-TP/assets/159874470/65842d09-6f27-46c7-ae71-e817982e1465)
4. LRP comparison
![image](https://github.com/jihyeonseong/SoM-TP/assets/159874470/4e46bfd8-7a4b-4a89-8308-54399a53d275)

## Citation
```
@article{Seong_Kim_Choi_2024, title={Towards Diverse Perspective Learning with Selection over Multiple Temporal Poolings}, volume={38}, url={https://ojs.aaai.org/index.php/AAAI/article/view/28743}, DOI={10.1609/aaai.v38i8.28743}, abstractNote={In Time Series Classification (TSC), temporal pooling methods that consider sequential information have been proposed. However, we found that each temporal pooling has a distinct mechanism, and can perform better or worse depending on time series data. We term this fixed pooling mechanism a single perspective of temporal poolings. In this paper, we propose a novel temporal pooling method with diverse perspective learning: Selection over Multiple Temporal Poolings (SoM-TP). SoM-TP dynamically selects the optimal temporal pooling among multiple methods for each data by attention. The dynamic pooling selection is motivated by the ensemble concept of Multiple Choice Learning (MCL), which selects the best among multiple outputs. The pooling selection by SoM-TP’s attention enables a non-iterative pooling ensemble within a single classifier. Additionally, we define a perspective loss and Diverse Perspective Learning Network (DPLN). The loss works as a regularizer to reflect all the pooling perspectives from DPLN. Our perspective analysis using Layer-wise Relevance Propagation (LRP) reveals the limitation of a single perspective and ultimately demonstrates diverse perspective learning of SoM-TP. We also show that SoM-TP outperforms CNN models based on other temporal poolings and state-of-the-art models in TSC with extensive UCR/UEA repositories.}, number={8}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Seong, Jihyeon and Kim, Jungmin and Choi, Jaesik}, year={2024}, month={Mar.}, pages={8948-8956} }
```
