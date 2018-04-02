FROM tensorflow/tensorflow:1.5.0-gpu-py3

RUN apt-get update && apt-get install -y python3-tk
RUN pip install tqdm


VOLUME /project
WORKDIR /project

CMD ["/bin/bash"]
