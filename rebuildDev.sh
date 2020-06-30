#!/bin/bash
docker rmi -f artyins-classifierservice
mkdir -p dockerdev
sudo rm -r dockerdev/artyins-classifierservice
rsync -r ../artyins-classifierservice dockerdev/
docker build ./dockerdev/. --no-cache -t artyins-classifierservice
