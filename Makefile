docker-run:
	docker run -ti --rm deepstan bash

docker-build:
	docker build -t deepstan -f deepstan.docker .