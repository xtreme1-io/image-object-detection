# Image Object Detection

Create base image:
```bash
cd base_image
docker build -t basicai/xtreme1-image-object-detection-base .
```

Using base container to run model service:
```bash
docker run -it --rm -p 5000:5000 -v .:/app --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -m 32G --memory-reservation 8G --cpu-shares=80 --shm-size=32G xtreme1-image-object-detection-base env LANG=C.UTF-8 /bin/bash

# The following commands run in the container
cd /app
wget https://basicai-asset.s3.us-west-2.amazonaws.com/xtreme1/model/coco80.pth

cd src
python -u server.py --device=0 --weights=../coco80.pt --conf-thres=0.5 --port=5000

# Test
python client_demo.py
```
