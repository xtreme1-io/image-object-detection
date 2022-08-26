enter the docker folder, choose a dockerfile corresponding to your cuda version, rename it to Dockerfile, and build an image:
sudo docker build -t custom_user/yolor:v1.0 .

create a container with image you just built:
sudo docker run -it -p 18881:18881 -v path_to_the_src_code:/home --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -m 32G --memory-reservation 8G --cpu-shares=80 --shm-size=32G --name custom_yolor custom_user/yolor:v1.0 env LANG=C.UTF-8 /bin/bash

to start inference server:
python -u server.py --device=0 --weights=../coco80.pt --conf-thres=0.5 --port=18881

to test the detection service with a web image:
python client_demo.py