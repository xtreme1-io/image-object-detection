FROM basicai/xtreme1-image-object-detection-base

RUN pip install flask_cors

WORKDIR /app
RUN wget https://basicai-asset.s3.us-west-2.amazonaws.com/xtreme1/model/coco80.pth
COPY . ./

WORKDIR /app/src
CMD ["python", "-u", "server.py", "--device=0", "--weights=../coco80.pt", "--conf-thres=0.5", "--port=5000"] 
