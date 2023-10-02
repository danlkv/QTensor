#!/bin/bash

docker build -f Dockerfile.dev -t dev .
docker run -v $(pwd):/app -it dev
