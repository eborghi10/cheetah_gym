#!/bin/bash
set -e

sudo chown -R $USER $HOME/catkin_ws
exec "$@"
