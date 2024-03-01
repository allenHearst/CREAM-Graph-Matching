# CREAM-Graph-Matching
CREAM Graph Matching Repository. See CREAM repository in [https://github.com/allenHearst/CREAM/].

# Get Started
We follow COMMON [https://github.com/XLearning-SCU/2023-ICCV-COMMON/] to build our codes. Please take a look at the instructions shown in the COMMON repository.

After creating a docker environment and ```git clone``` COMMON, please use the codes shown in this repository to replace these files:
```
|--models
|  |--COMMON
|     |--model.py
|--src
|  |--dataset
|     |--data_loader.py
|  |--loss_func.py
|--eval.py
|--train_eval.py
```
Then you can follow the instructions shown in COMMON repository to start training or testing.

# Citation
If you found our work useful, please cite this work as follows, thank you.
```
@article{ma2024cream,
	title={Cross-modal Retrieval with Noisy Correspondence via Consistency Refining and Mining},
	author={Ma, Xinran and Yang, Mouxing and Li, Yunfan and Hu, Peng and Lv, Jiancheng and Peng, Xi},
	journal={IEEE transactions on image processing},
	year={2024}
}
```
