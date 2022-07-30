# algo-runner
![workflow](https://github.com/achains/algo-runner/actions/workflows/black.yml/badge.svg)
## Abstract 
There are many algoritmhs which are used for plane segmentation, especially based on RANSAC technique. Despite the high accuracy given by RANSAC, the method has its own limitations.

There is class of plane segmentation algorithms which are tuned to work with data obtained from RGB-D Camera. The existing implemtation of most of them is written in C++, so it may be hard to build and configure each of the algorithms.

**plane-seg** is meant to be an easy-to-use wrapper for running and configuring those algorithms. To run algorithm we use Docker images from [3D-plane-segmentation](https://github.com/MobileRoboticsSkoltech/3D-plane-segmentation). More than that we provide interface for evaluating the results of segmentation based on [EVOPS package](https://pypi.org/project/evops/).
## External dependencies
All Python dependencies are specified in `python/requirements.txt` and can be installed with *pip* or another package manager.

If you want to run any of the wrapped algorithms, you need to have applicable Docker image installed on your system:

| Algorithm | Docker image |
| ------ | ------ |
| DDPFF | [image](https://github.com/MobileRoboticsSkoltech/3D-plane-segmentation/tree/main/DDPFF) |
| peac | [image](https://github.com/MobileRoboticsSkoltech/3D-plane-segmentation/tree/main/peac) |
| CAPE | [image](https://github.com/MobileRoboticsSkoltech/3D-plane-segmentation/tree/main/CAPE) |

## Examples
You may find examples of running the algorithm in `python/examples` as well as evaluating metrics on segmentation result.

## License
Apache License, Version 2.0
