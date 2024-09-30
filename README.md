# MVInpainter
[NeurIPS 2024] MVInpainter: Learning Multi-View Consistent Inpainting to Bridge 2D and 3D Editing

[[arXiv]](https://arxiv.org/pdf/2408.08000) [[Project Page]](https://ewrfcas.github.io/MVInpainter/)

Codes and models will be open-released soon!

Codes organization is working in progress.

### TODO List
- [ ] Environment and dataset setup
- [ ] Training codes
- [ ] Inference models and pipeline

## Preparation

### Setup repository and environment
```
git clone https://github.com/ewrfcas/MVInpainter.git
cd MVInpainter

conda create -n mvinpainter python=3.8
conda activate mvinpainter

pip install -r requirements.txt
mim install mmcv-full
pip install mmflow

# We need to replace the new decoder py of mmflow for faster flow estimation
cp ./check_points/mmflow/raft_decoder.py /usr/local/conda/envs/mvinpainter/lib/python3.8/site-packages/mmflow/models/decoders/
```


## Cite
If you found our program helpful, please consider citing:

```
@article{cao2024mvinpainter,
  title={MVInpainter: Learning Multi-View Consistent Inpainting to Bridge 2D and 3D Editing},
  author={Cao, Chenjie and Yu, Chaohui and Fu, Yanwei and Wang, Fan and Xue, Xiangyang},
  journal={arXiv preprint arXiv:2408.08000},
  year={2024}
}
```