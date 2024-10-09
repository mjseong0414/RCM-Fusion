# resnet101 train & test
./tools/dist_train.sh ./projects/configs/rcmfusion_icra/rcm-fusion_r101.py 4 --work-dir ./workdirs/rcm_fusion_r101
./tools/dist_test.sh ./projects/configs/rcmfusion_icra/rcm-fusion_r101.py ./ckpts/rcm-fusion-r101-icra-final.pth 4 --eval bbox

# resnet50 train & test
./tools/dist_train.sh ./projects/configs/rcmfusion_icra/rcm-fusion_r50.py 4 --work-dir ./workdirs/rcm_fusion_r50/
./tools/dist_test.sh ./projects/configs/rcmfusion_icra/rcm-fusion_r50.py ./ckpts/rcm-fusion-r50-icra-final.pth 4 --eval bbox