#!/bin/bash
for i in 0 1
do
    for j in 0 1
    do
        for k in 0 1 
        do 
            for l in 0 1
            do 
                python3 td3.py --mode 1 --hid 256 --l 2 --n $i --psn $j --decay $k --layernorm $l
            done
        done
    done
done