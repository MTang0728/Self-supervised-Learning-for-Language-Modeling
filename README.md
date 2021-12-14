# Comparing Self-supervised Learning Pretext Tasks for Language Modeling 

## Abstract
Self-supervised learning is rapidly gaining popularity. It enables the model to learn from a large number of unlabeled datasets, and transfer the knowledge learned to perform specific downstream tasks. Self-supervised learning generally has two main stages. The first Pre-training stage trains the model on a large general-purpose unlabeled dataset with various pretext tasks. The second Fine-tuning stage uses the weights learned in the pre-training stage as initial weights and trains the model on a smaller labeled dataset to perform specific tasks. In this project, we compared the impact of different pretext tasks on a downstream task in language modeling. Ablation studies were also implemented to compare the marginal effect of each pretext task and the interaction between pretext tasks. We found that Masked Language Modeling (MLM) performs the best, achieving 0.86 test accuracy; Sentence Order Prediction (SOP) produced the lowest accuracy at 0.77 on the IMDB dataset. We also found that the performance on the pre-training phase is not an indicator of the performance on the specific downstream task. 

## Notebooks Organization
- All model checkpoints can be found in the 'pretrain' and 'finetune' folders 
#### 1) Three Layer Transformer Encoder
    - attention.py
    - transformer.py
#### 1) MLM
    - Pre-train & Finetune: mlm.ipynb
#### 2) NSP
    - Pre-train: pretext_task_pretrain.ipynb
    - Fine-tune: finetune_nsp.ipynb
#### 3) SOP
    - Pre-train: pretext_task_pretrain.ipynb
    - Fine-tune: finetune_sop.ipynb
#### 4) SP
    - Pre-train & Finetune: sp.ipynb
#### 5) MLM + NSP
    - Pre-train & Finetune: mlm_nsp_sop.ipynb
#### 6) MLM + SOP
    - Pre-train & Finetune: mlm_nsp_sop.ipynb
#### 7) MLM + SP
    - Pre-train & Finetune: mlm_sp.ipynb
## Results:
Pre-train Tasks | Best Validation Prediction Accuracy | Sentiment Classification Test Accuracy
-----|-------|-----
Baseline | - | 0.861
MLM|0.035 | 0.860
NSP|0.578 | 0.858
SOP|0.510 | 0.773
SP|0.250 | 0.851
MLM + NSP|0.297 | 0.849
MLM + SOP|0.273 | 0.861
MLM + SP|0.249 | 0.851