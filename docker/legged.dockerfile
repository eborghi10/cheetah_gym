FROM robotica:ros1_noetic

USER root

# APT Dependencies
COPY dependencies.legged dependencies.legged
RUN apt-get update && \
    xargs -a dependencies.legged apt-get install -y -qq

COPY requirements.legged.txt requirements.legged.txt
RUN cat requirements.legged.txt | xargs pip3 install

# Install gym
RUN git clone https://github.com/openai/gym.git \
    && cd gym \
    && pip install -e .

# Adding this path is needed because the virtual environment
# overwrites the path and some ROS modules cannot be found
# ENV PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}"

CMD [ "/bin/bash", "-c" ]
