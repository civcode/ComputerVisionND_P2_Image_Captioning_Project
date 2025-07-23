docker run \
    --gpus all \
    --runtime=nvidia \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=8g \
    -it --rm \
    -p 8888:8888 \
    -u $(id -u):$(id -g) \
    -e HOME=$HOME \
    -e USER=$USER \
    -v /home/$USER:/home/$USER \
    -v /home/$USER/workspace:/workspace \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    pytorch-cuda-docker
