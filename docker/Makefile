# Base images

.PHONY: ubuntu_18
ubuntu_18:
	docker build \
		--build-arg USER_ID=$(shell id -u) \
		--build-arg GROUP_ID=$(shell id -g) \
		--build-arg UBUNTU=18 \
		-f ubuntu.dockerfile \
		-t robotica:ubuntu_18 \
		.

.PHONY: ubuntu_20
ubuntu_20:
	docker build \
		--build-arg USER_ID=$(shell id -u) \
		--build-arg GROUP_ID=$(shell id -g) \
		--build-arg UBUNTU=20 \
		-f ubuntu.dockerfile \
		-t robotica:ubuntu_20 \
		.

# ROS 1 images

.PHONY: ros1_noetic
ros1_noetic: ubuntu_20
	docker build \
		--build-arg UBUNTU=20 \
		--build-arg ROS=noetic \
		-f ros1.dockerfile \
		-t robotica:ros1_noetic \
		.


# Robotics images

.PHONY: ekuthon_2021
ekuthon_2021: ros1_noetic
	docker build \
		-f legged.dockerfile \
		-t ekuthon:2021 \
		.
