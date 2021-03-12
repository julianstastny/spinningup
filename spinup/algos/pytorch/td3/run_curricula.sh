#!/bin/bash
for i in 1 5
do
    for j in 0 1
    do
        for k in 0 1 
        do 
            for l in 0 1
            do 
                python3 td3.py --env HockeyCurriculum-v0 --hid 256 --l 2 --n $i --psn $j --decay $k --layernorm $l > output.txt
            done
        done
    done
done