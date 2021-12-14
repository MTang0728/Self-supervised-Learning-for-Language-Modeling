# Comparing Self-supervised Learning Pretext Tasks for Language Modeling 

## Abstract
Self-supervised learning is rapidly gaining popularity. It enables the model to learn from a large number of unlabeled datasets, and transfer the knowledge learned to perform specific downstream tasks. Self-supervised learning generally has two main stages. The first Pre-training stage trains the model on a large general-purpose unlabeled dataset with various pretext tasks. The second Fine-tuning stage uses the weights learned in the pre-training stage as initial weights and trains the model on a smaller labeled dataset to perform specific tasks. In this project, we compared the impact of different pretext tasks used in BERT, ALBERT and BART on Sentiment Analysis. Ablation studies were also implemented to compare the marginal effect of each pretext task and the interaction between pretext tasks. We found that Masked Language Modeling (MLM) performs the best, achieving 0.86 test accuracy; Sentence Order Prediction (SOP) produced the lowest accuracy at 0.77 on the IMDB dataset. We also found that the performance on the pre-training phase is not an indicator of the performance on the specific downstream task. 

## Results:
Pre-train Tasks | Best Validation Prediction Accuracy | Sentiment Classification Test Accuracy | Original Model
-----|-------|-----|-----
Baseline | - | 0.861 | Transformer
MLM|0.035 | 0.860 | BERT
NSP|0.578 | 0.858 | BERT
SOP|0.510 | 0.773 | ALBERT
SP|0.250 | 0.851 | BART
MLM + NSP|0.297 | 0.849 | -
MLM + SOP|0.273 | 0.861 | -
MLM + SP|0.249 | 0.851 | -

Please find the report that covers our implementation detail and results [here](https://github.com/MTang0728/Self-supervised-Learning-for-Language-Modeling/blob/main/docs/report.pdf).

You may also find summary of our project in a poster form [here](https://github.com/MTang0728/Self-supervised-Learning-for-Language-Modeling/blob/main/docs/poster.pdf).

## Team Members
Dean Huang: [@DeanHuang-Git](https://github.com/DeanHuang-Git)   <br />
Michael Tang: [@MTang0728](https://github.com/MTang0728)   <br />
Betty Wu: [@JiamanBettyWu](https://github.com/JiamanBettyWu)   <br />
