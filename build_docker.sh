TAG=${1:-latest}

docker build -t movienet:$TAG .
