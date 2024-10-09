# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train RCM-Fusion with 4 GPUs 
```
./tools/dist_train.sh ./projects/configs/rcmfusion_icra/rcm-fusion_r101.py 4 --work-dir ./workdirs/rcm_fusion_r101
```

Eval RCM-Fusion with 4 GPUs
```
./tools/dist_test.sh ./projects/configs/rcmfusion_icra/rcm-fusion_r101.py ./ckpts/rcm-fusion-r101-icra-final.pth 4 --eval bbox
```