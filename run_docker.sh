TAG=${1:-latest}

docker run --rm -it --gpus all movienet:$TAG
# Then run python3 demos/audio_demo.py
