# https://hub.docker.com/r/nvidia/cudagl
ARG UBUNTU
FROM nvidia/cudagl:11.3.0-runtime-ubuntu${UBUNTU}.04

LABEL maintainer="Emiliano Borghi <eborghiorue@frba.utn.edu.ar>"

ARG USER_ID
ARG GROUP_ID
ENV USER=robotica

# Setup environment
RUN apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV \
  LANG=en_US.UTF-8 \
  DEBIAN_FRONTEND=noninteractive \
  TERM=xterm

# Common dependencies
COPY dependencies.common dependencies.common
RUN apt-get update && \
    xargs -a dependencies.common apt-get install -y -qq

# Create a user with passwordless sudo
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --gecos '' --disabled-password --uid $USER_ID --gid $GROUP_ID $USER
RUN adduser $USER sudo
RUN echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
