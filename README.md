
# Data Selection Matters: Towards Robust Instruction Tuning of Large Multimodal Models

This is the code repository for paper [Data Selection Matters: Towards Robust Instruction
Tuning of Large Multimodal Models](https://openreview.net/forum?id=5CFhUQCmkW), NIPS 2025.

###  Abstract
Selecting a compact subset of visual instruction–following data has emerged as an effective way to align large multimodal models with human intentions while avoiding the high cost of full-dataset training. Yet we observe that both full-data training and existing state-of-the-art data selection methods tend to inherit underlying dataset biases such as position bias and spurious correlations, leading to biased model behaviors. To address this issue, we introduce ARDS, a robustness-aware targeted visual instruction-selection framework that explicitly mitigates these weaknesses, sidestepping the need for access to downstream data or time-consuming gradient computation. Specifically, we first identify the worst-case evaluation subgroups through visual and textual task-specific perturbations. The robust training  mixture is then constructed by prioritizing samples that are semantically closer to  these subgroups in a rich multimodal embedding space. Extensive experiments demonstrate that ARDS substantially boosts both robustness and data efficiency for  visual instruction tuning. We also showcase that the robust mixtures produced with  a smaller model transfer effectively to larger architectures.



### Environment Preparation
Clone this repository,
```
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
The code was tested with Python 3.10.0 and Pytorch >= 2.1


### Data Preparation
Refer to LLaVA for downloading the annotation of the visual instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), we save all files as `.jpg`
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

### Selection
```bash
bash selection.sh
```

### Training
First download the pre-trained connector.
```bash
 # for llava-v1.5-7b
git clone https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5

 # for llava-v1.5-13b
git clone https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5
```
Refer to [models](docs/MODEL_ZOO.md) for more choices.

Start training with different models. Set the data path to full-data and selected data.
```bash
 # for full-model training
bash finetune_7b.sh

 # for LoRA training
bash finetune_lora.sh

 # for 13b training
bash finetune.sh
```

### Test on clean input and PA/SA attacks
```
bash scripts/v1_5/eval/sqa.sh
bash scripts/v1_5/eval/seed.sh
```
See more evaluation in [Evaluation](docs/Evaluation.md)

### Trained Models and Selected Results
```bash
# full-data full-model trained
https://huggingface.co/liuhaotian/llava-v1.5-7b

# full-data lora trained
https://drive.google.com/drive/folders/1KBgiB4AcvIgUkfTXs_wiZSWXzMaIxRVN?usp=drive_link

# ards lora trained
https://drive.google.com/drive/folders/1VvRi-x61GJ0UXmXUYjsiyIYvD744qaZ2?usp=drive_link

# ards selected results
https://drive.google.com/file/d/1rgzC3-aO-AgX08452HrlyxHWnldrjm4o/view?usp=sharing
```

### Citation
If you find our project helpful, please consider cite our paper:
```
@inproceedings{robustmixture25,
title={Data Selection Matters: Towards Robust Instruction Tuning of Large Multimodal Models},
author={Yang, Xu and Liu, Chen and Wei, Ying},
booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
year={2025},
}
```

### Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [LESS](https://github.com/princeton-nlp/LESS)
