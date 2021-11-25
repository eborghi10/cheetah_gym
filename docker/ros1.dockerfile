ARG UBUNTU
FROM robotica:ubuntu_${UBUNTU}

ARG ROS
ENV ROS_DISTRO=${ROS}

# Install ROS
# Setup sources.list for ROS
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
# Setup keys for ROS
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# Install ROS packages
COPY dependencies.ros1_$ROS_DISTRO dependencies.ros1_$ROS_DISTRO
RUN apt-get update && \
  xargs -a dependencies.ros1_$ROS_DISTRO apt-get install -y -qq

# Automatically source ROS workspace
RUN echo ". /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USER}/.bashrc
ENV CATKIN_SETUP_BASH "/home/${USER}/catkin_ws/devel/setup.bash"
RUN echo "[[ -f ${CATKIN_SETUP_BASH} ]] && . ${CATKIN_SETUP_BASH}" >> /home/${USER}/.bashrc

# Initialize rosdep
RUN rosdep init
USER $USER
# HOME needs to be set explicitly. Without it, the HOME environment variable is set to "/"
RUN HOME=/home/$USER rosdep update

######################################################

# setup entrypoint
USER root
COPY ./ros1_entrypoint.sh /
RUN chmod +x /ros1_entrypoint.sh

ENTRYPOINT [ "/ros1_entrypoint.sh" ]
CMD [ "/bin/bash", "-c" ]
