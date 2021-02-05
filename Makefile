all:
	head -n 18 README.md

init:
	git submodule init && git submodule update
	pip install ./posteriordb/python
	opam pin -y -k git git+https://github.com/deepppl/stanc3.git
	pip install -r requirements.txt

docker-run:
	docker run -ti --rm deepstan bash

docker-build:
	docker build -t deepstan -f deepstan.docker .

clean:
	-find example-models -name "*.py" -exec rm {} \;

cleanall: clean
	-rm -rf rq1/*.csv rq1/_tmp logs-*
