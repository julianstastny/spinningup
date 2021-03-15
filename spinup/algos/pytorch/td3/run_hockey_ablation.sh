python3 td3.py --env Hockey-One-v0 --mode 0 --hid 256 --l 2 --n 3 --psn 1 --decay 1 --layernorm 1 --epochs 50 --seed 0 --exp_name weakAblation  

python3 td3.py --env Hockey-One-v0 --mode 0 --hid 256 --l 2 --n 1 --psn 1 --decay 1 --layernorm 1 --epochs 50 --seed 0 --exp_name weakAblation  

python3 td3.py --env Hockey-One-v0 --mode 0 --hid 256 --l 2 --n 3 --psn 0 --decay 1 --layernorm 1 --epochs 50 --seed 0 --exp_name weakAblation  

python3 td3.py --env Hockey-One-v0 --mode 0 --hid 256 --l 2 --n 3 --psn 1 --decay 0 --layernorm 1 --epochs 50 --seed 0 --exp_name weakAblation  

python3 td3.py --env Hockey-One-v0 --mode 0 --hid 256 --l 2 --n 3 --psn 1 --decay 1 --layernorm 0 --epochs 50 --seed 0 --exp_name weakAblation  