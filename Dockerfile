FROM registry.talos.basic.ai/basicai/algorithm/images/env/yolor:latest

# install aws cli and config
RUN pip install awscli \
    && mkdir -p ~/.aws \
    && printf "[default]\naws_access_key_id = ${BASICAI_ALGORITHM_AWS_ACCESS_KEY_READ}\naws_secret_access_key = ${BASICAI_ALGORITHM_AWS_ACCESS_SECRET_READ}\n" > ~/.aws/credentials

WORKDIR /home

ENV PATH="/home/.local/bin:${PATH}"
COPY ./src /home

# download model file
RUN aws s3 cp ${MODEL_FILE_S3} /home/best_overall.pt

WORKDIR /home
EXPOSE 5000
ENTRYPOINT python -u server_abroad.py --device=0 --weights=best_overall.pt --conf-thres=0.5 --port=5000
