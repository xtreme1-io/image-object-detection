FROM  basicai/xtreme1-image-object-detection:base
RUN pip install flask_cors
COPY ./ /yolor
WORKDIR /yolor/src
CMD ["python","-u","server.py","--device=0","--weights=../coco80.pt","--conf-thres=0.5","--port=5000"] 
