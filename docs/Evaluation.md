# Evaluation


First download official evaluation scripts [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing). It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Extract to `./playground/data/eval`. This also provides a general structure for all datasets.

### ScienceQA

1. Under `./playground/data/eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh
```

### SEED-Bench

1. Following the official [instructions](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md) to download the images and the videos. Put images under `./playground/data/eval/seed_bench/SEED-Bench-image`.
2. Extract the video frame in the middle from the downloaded videos, and put them under `./playground/data/eval/seed_bench/SEED-Bench-video-image`. We provide our script `extract_video_frames.py` modified from the official one.
3. Multiple-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/seed.sh
```
4. Optionally, submit the results to the leaderboard: `./playground/data/eval/seed_bench/answers_upload` using the official jupyter notebook.


### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `./playground/data/eval/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench.sh
```
3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712`.

### MMBench-CN

1. Download [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) and put under `./playground/data/eval/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench_cn.sh
```
3. Submit the results to the evaluation server: `./playground/data/eval/mmbench/answers_upload/mmbench_dev_cn_20231003`.


### GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions and put under `./playground/data/eval/gqa/data`. You may need to modify `eval.py` as [this](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187) due to the missing assets in the GQA v1.2 release.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/gqa.sh
```

### GQA-OOD
See `./playground/data/eval/gqa-ood/data`. We provide `eval.py`.



### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `./playground/data/eval/textvqa`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh
```
