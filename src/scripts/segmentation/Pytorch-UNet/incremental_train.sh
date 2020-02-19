python3 train.py -s 1 -e 2 -l 1e-6 -v 20.0 -f checkpoints/CP_epoch2.pth
cp checkpoints/CP_epoch1.pth weights/CP_epoch1_1e-6.pth
cp checkpoints/CP_epoch2.pth weights/CP_epoch2_1e-6.pth
python3 train.py -s 1 -e 2 -l 1e-7 -v 20.0 -f checkpoints/CP_epoch2.pth
cp checkpoints/CP_epoch1.pth weights/CP_epoch1_1e-7.pth
cp checkpoints/CP_epoch2.pth weights/CP_epoch2_1e-7.pth
python3 train.py -s 1 -e 2 -l 1e-8 -v 20.0 -f checkpoints/CP_epoch2.pth
cp checkpoints/CP_epoch1.pth weights/CP_epoch1_1e-8.pth
cp checkpoints/CP_epoch2.pth weights/CP_epoch2_1e-8.pth
