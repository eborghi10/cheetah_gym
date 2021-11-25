FROM robotica:ros1_noetic

USER root

# APT Dependencies
COPY dependencies.legged dependencies.legged
RUN apt-get update && \
    xargs -a dependencies.legged apt-get install -y -qq

# Install ROS packages from source
ENV LEGGED_WS=/legged_ws
RUN mkdir -p ${LEGGED_WS}/src
WORKDIR ${LEGGED_WS}/src
RUN apt-get install -yqq python3-catkin-tools python3-vcstool
COPY legged.repos ${LEGGED_WS}/src/legged.repos
RUN vcs import . < legged.repos --recursive
WORKDIR $LEGGED_WS
# Remove CMake package
RUN rm -rf ${LEGGED_WS}/src/towr/towr
RUN apt-get update
USER $USER
RUN rosdep install --from-paths src --rosdistro=${ROS_DISTRO} -yi -r
USER root
RUN /bin/bash -c ". /opt/ros/${ROS_DISTRO}/setup.bash; \
    catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ros/${ROS_DISTRO}; \
    cd build; make install"
RUN rm -r ${LEGGED_WS}

COPY requirements.legged.txt requirements.legged.txt
RUN cat requirements.legged.txt | xargs pip3 install

# Install a virtual environment
RUN apt-get update && apt-get install -y python3.8-venv && apt-get clean all
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install gym
RUN git clone https://github.com/openai/gym.git \
    && cd gym \
    && pip install -e .
# Install pybullet-gym
RUN git clone https://github.com/benelot/pybullet-gym.git \
    && cd pybullet-gym \
    && pip install -e .

# Install elevation_mapping packages with catkin_make_isolated
ENV LEGGED_ISOLATED_WS=/legged_isolated_ws
RUN mkdir -p ${LEGGED_ISOLATED_WS}/src
COPY legged_isolated.repos legged_isolated.repos
RUN vcs import $LEGGED_ISOLATED_WS/src < legged_isolated.repos --recursive
WORKDIR $LEGGED_ISOLATED_WS
RUN apt-get update -qq
USER $USER
RUN rosdep install --from-paths src --rosdistro=${ROS_DISTRO} -yi -r
USER root
RUN /bin/bash -c ". /opt/ros/${ROS_DISTRO}/setup.bash; \
    catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release \
                        -DCMAKE_INSTALL_PREFIX=/opt/ros/${ROS_DISTRO} \
                        -DPYTHON_EXECUTABLE=/usr/bin/python3"
RUN rm -r ${LEGGED_ISOLATED_WS}

# Adding this path is needed because the virtual environment
# overwrites the path and some ROS modules cannot be found
ENV PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}"

CMD [ "/bin/bash", "-c" ]
