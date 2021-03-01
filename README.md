[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mutual-modality-learning-for-video-action/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=mutual-modality-learning-for-video-action)

# Mutual Modality Learning for Video Action Classification

By Stepan Komkov, Maksim Dzabraev and Aleksandr Petiushko

This is the code for the [Mutual Modality Learning article](https://arxiv.org/abs/2011.02543).

## Abstract

The construction of models for video action classification progresses rapidly. 
However, the performance of those models can still be easily improved by ensembling 
with the same models trained on different modalities (e.g. Optical flow). Unfortunately, 
it is computationally expensive to use several modalities during inference. Recent works 
examine the ways to integrate advantages of multi-modality into a single RGB-model. Yet, 
there is still a room for improvement. In this paper, we explore the various methods to 
embed the ensemble power into a single model. We show that proper initialization, as well 
as mutual modality learning, enhances single-modality models. As a result, we achieve 
state-of-the-art results in the Something-Something-v2 benchmark.

## Code changes

This is a forked repository
from the [code](https://github.com/mit-han-lab/temporal-shift-module) presented 
in the [TSM article](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.pdf).

Since this is a forked repository, we describe only differences in regard to the
[original code](https://github.com/mit-han-lab/temporal-shift-module).

### Mutual Learning and Mutual Modality Learning launch

In order to launch ordinary Mutual Learning with RGB inputs use commands

```
python main.py somethingv2 RGB,RGB --rank 0 --world_size 2 [other training parameters]
python main.py somethingv2 RGB,RGB --rank 1 --world_size 2 [other training parameters]
```

In order to launch Mutual Mutual Learning with RGB-, Flow- and Diff-based models use commands

```
python main.py somethingv2 RGBDiff,Flow,RGB --rank 0 --world_size 3 [other training parameters]
python main.py somethingv2 RGBDiff,Flow,RGB --rank 1 --world_size 3 [other training parameters]
python main.py somethingv2 RGBDiff,Flow,RGB --rank 2 --world_size 3 [other training parameters]
```

Use `--gpus` and `--init_method` arguments to specify devices for each model and/or launch multi-node training.

Use `--tune_from` argument to specify the initialization model (the same way as before).

Use `--random_sample` argument to turn on random sampling strategy during training.

Use `--dense_length` argument to specify the number of frames for the dense sampling (it also affects random sampling).

Thus, these are the minimum commands to reproduce MML results:

```
python main.py somethingv2 RGB [other training parameters]

python main.py somethingv2 RGB,Flow --rank 0 --world_size 2 --tune_from $PATH_TO_MODEL_FROM_THE_FIRST_STEP$ [other training parameters]
python main.py somethingv2 RGB,Flow --rank 1 --world_size 2 --tune_from $PATH_TO_MODEL_FROM_THE_FIRST_STEP$ [other training parameters]
```

### Testing part

The testing script can be launched as before. There are several new functions available during testing.

Use `--random_sample` argument to use both uniform and dense sampling during testing.

Use `--dense_length` argument to specify the number of frames for the dense sampling (it also affects random sampling).

Use `--dense_number` argument to specify the number of dense samplings (it also affects random sampling).

Use `--twice_sample` argument to use two uniform samplings during testings that are shifted by the half-period (it also affects random sampling).

## Citation

```
@article{komkov2020mml,
  title={Mutual Modality Learning for Video Action Classification},
  author={Komkov, Stepan and Dzabraev, Maksim and Petiushko, Aleksandr},
  journal={arXiv preprint arXiv:2011.02543},
  year={2020}
}
```
