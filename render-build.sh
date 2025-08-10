#!/usr/bin/env bash

pip install -r requirements.txt

mkdir -p ffmpeg-bin
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | \
tar -xJ --strip-components=1 -C ffmpeg-bin --wildcards '*/ffmpeg' '*/ffprobe'