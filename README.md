# TDPS-Arrow-Recognition
Recognition on mobile or embedded devices.
## Installation
- CUDA/Python
## Quick Start
(1) Collect the ARROW dataset to `/your/path`.

(2) Train and test different models.
```python
python train.py --network "CompactNet" --root "/your/path" --epoch 15 --base_lr 1e-3 --batchsize 64
```
Note: change `--network "CompactNet"` for training on different models.  
Note: change `--root "/your/path"` to your data path.  
