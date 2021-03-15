python3 td3.py --env Hockey-One-v0 --mode 0 --weak_opponent 1 --epochs 100 --psn 1 --decay 1 --layernorm 1 --exp_name vsWeak

python3 td3.py --env Hockey-One-v0 --mode 0 --weak_opponent 0 --epochs 100 --psn 1 --decay 1 --layernorm 1 --exp_name vsStrong

python3 td3.py --env HockeyCurriculum-v0 --epochs 100 --psn 1 --decay 1 --layernorm 1 --self_play 0 --exp_name noSPCUrr