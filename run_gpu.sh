docker run \
--cpus="2" \
--memory=4g \
--gpus all \
--rm \
-it \
-v /home/paolo/code_base/ffserver/data:/usr/src/app/inference/video \
pmonteverdi/polifemo:v1.0