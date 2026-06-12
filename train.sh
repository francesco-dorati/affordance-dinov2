conda activate robotics-affordance
rm -rf checkpoints
python scripts/train.py —epochs 25
python scripts/evaluate.py --checkpoint checkpoints/best.pth  

