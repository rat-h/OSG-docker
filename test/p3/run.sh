#! /bin/bash

docker run -ti -p 8888:8888 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/home -m 8g $1 python network.py /gui=False preview=False git=False /neuron/Type=2 /tstop=200 /neuron/Iapp=2e-2
