all: scaled

.PHONY: eval scaled eval_accuracy eval_speed scaled_accuracy scaled_speed clean cleanall

eval: eval_accuracy eval_speed

scaled: scaled_accuracy scaled_speed

eval_accuracy:
	python test_accuracy.py --backend stan
	python test_accuracy.py --backend pyro --mode comprehensive
	python test_accuracy.py --backend numpyro --mode comprehensive
	python test_accuracy.py --backend numpyro --mode mixed
	python test_accuracy.py --backend numpyro --mode generative

eval_speed:
	python test_speed.py --backend stan
	python test_speed.py --backend pyro --mode comprehensive
	python test_speed.py --backend numpyro --mode comprehensive
	python test_speed.py --backend numpyro --mode mixed
	python test_speed.py --backend numpyro --mode generative

scaled_accuracy:
	python test_accuracy.py --backend stan --scaled
	python test_accuracy.py --backend numpyro --mode comprehensive --scaled
	python test_accuracy.py --backend numpyro --mode mixed --scaled
	python test_accuracy.py --backend numpyro --mode generative --scaled

scaled_speed:
	python test_speed.py --backend stan --scaled
	python test_speed.py --backend numpyro --mode comprehensive --scaled
	python test_speed.py --backend numpyro --mode mixed --scaled
	python test_speed.py --backend numpyro --mode generative --scaled

clean:
	rm -rf _tmp __pycache__ _build_*

cleanall: clean
	rm -rf logs *~
