
CUDA_VISIBLE_DEVICES=0 bash eval.sh tests_unseen 0 64 flare &
CUDA_VISIBLE_DEVICES=0 bash eval.sh tests_unseen 191 255 flare &
CUDA_VISIBLE_DEVICES=0 bash eval.sh tests_unseen 64 127 flare &
CUDA_VISIBLE_DEVICES=0 bash eval.sh tests_unseen 255 319 flare &
CUDA_VISIBLE_DEVICES=0 bash eval.sh tests_unseen 127 191 flare &
CUDA_VISIBLE_DEVICES=0 bash eval.sh tests_unseen 319 382 flare &


CUDA_VISIBLE_DEVICES=1 bash eval.sh tests_unseen 382 446 flare &
CUDA_VISIBLE_DEVICES=1 bash eval.sh tests_unseen 573 637 flare &
CUDA_VISIBLE_DEVICES=1 bash eval.sh tests_unseen 446 510 flare &
CUDA_VISIBLE_DEVICES=1 bash eval.sh tests_unseen 637 701 flare &
CUDA_VISIBLE_DEVICES=1 bash eval.sh tests_unseen 510 573 flare &
CUDA_VISIBLE_DEVICES=1 bash eval.sh tests_unseen 701 764 flare &




CUDA_VISIBLE_DEVICES=2 bash eval.sh tests_unseen 764 828 flare &
CUDA_VISIBLE_DEVICES=2 bash eval.sh tests_unseen 956 1019 flare &
CUDA_VISIBLE_DEVICES=2 bash eval.sh tests_unseen 828 892 flare &
CUDA_VISIBLE_DEVICES=2 bash eval.sh tests_unseen 1019 1083 flare &
CUDA_VISIBLE_DEVICES=2 bash eval.sh tests_unseen 892 956 flare &
CUDA_VISIBLE_DEVICES=2 bash eval.sh tests_unseen 1083 1147 flare &



CUDA_VISIBLE_DEVICES=3 bash eval.sh tests_unseen 1147 1210 flare & 
CUDA_VISIBLE_DEVICES=3 bash eval.sh tests_unseen 1338 1402 flare &
CUDA_VISIBLE_DEVICES=3 bash eval.sh tests_unseen 1210 1274 flare & 
CUDA_VISIBLE_DEVICES=3 bash eval.sh tests_unseen 1402 1465 flare & 
CUDA_VISIBLE_DEVICES=3 bash eval.sh tests_unseen 1274 1338 flare & 
CUDA_VISIBLE_DEVICES=3 bash eval.sh tests_unseen 1465 1529 flare & 




CUDA_VISIBLE_DEVICES=4 bash eval.sh tests_seen 0 64 flare &
CUDA_VISIBLE_DEVICES=4 bash eval.sh tests_seen 64 128 flare &
CUDA_VISIBLE_DEVICES=4 bash eval.sh tests_seen 128 192 flare &
CUDA_VISIBLE_DEVICES=4 bash eval.sh tests_seen 192 256 flare &
CUDA_VISIBLE_DEVICES=4 bash eval.sh tests_seen 256 319 flare &
CUDA_VISIBLE_DEVICES=4 bash eval.sh tests_seen 319 383 flare &

CUDA_VISIBLE_DEVICES=5 bash eval.sh tests_seen 383 447 flare &
CUDA_VISIBLE_DEVICES=5 bash eval.sh tests_seen 447 511 flare &
CUDA_VISIBLE_DEVICES=5 bash eval.sh tests_seen 511 575 flare &
CUDA_VISIBLE_DEVICES=5 bash eval.sh tests_seen 575 639 flare &
CUDA_VISIBLE_DEVICES=5 bash eval.sh tests_seen 639 703 flare &
CUDA_VISIBLE_DEVICES=5 bash eval.sh tests_seen 703 766 flare &

CUDA_VISIBLE_DEVICES=6 bash eval.sh tests_seen 766 830 flare &
CUDA_VISIBLE_DEVICES=6 bash eval.sh tests_seen 830 894 flare &
CUDA_VISIBLE_DEVICES=6 bash eval.sh tests_seen 894 958 flare &
CUDA_VISIBLE_DEVICES=6 bash eval.sh tests_seen 958 1022 flare &
CUDA_VISIBLE_DEVICES=6 bash eval.sh tests_seen 1022 1086 flare &
CUDA_VISIBLE_DEVICES=6 bash eval.sh tests_seen 1086 1150 flare &

CUDA_VISIBLE_DEVICES=7 bash eval.sh tests_seen 1150 1214 flare &
CUDA_VISIBLE_DEVICES=7 bash eval.sh tests_seen 1214 1278 flare &
CUDA_VISIBLE_DEVICES=7 bash eval.sh tests_seen 1278 1341 flare &
CUDA_VISIBLE_DEVICES=7 bash eval.sh tests_seen 1341 1405 flare &
CUDA_VISIBLE_DEVICES=7 bash eval.sh tests_seen 1405 1469 flare &
CUDA_VISIBLE_DEVICES=7 bash eval.sh tests_seen 1469 1533 flare && fg
