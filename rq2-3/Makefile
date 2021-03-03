eval: test_accuracy.py test_speed.py utils.py
	python test_accuracy.py --backend stan 
	python test_accuracy.py --backend pyro --mode comprehensive 
	python test_accuracy.py --backend numpyro --mode comprehensive 
	python test_accuracy.py --backend numpyro --mode mixed 
	python test_accuracy.py --backend numpyro --mode generative 
	python test_speed.py --backend stan 
	python test_speed.py --backend pyro --mode comprehensive 
	python test_speed.py --backend numpyro --mode comprehensive 
	python test_speed.py --backend numpyro --mode mixed 
	python test_speed.py --backend numpyro --mode generative
	
scaled: test_accuracy.py test_speed.py utils.py
	python test_accuracy.py --backend stan --scaled 
	python test_accuracy.py --backend numpyro --mode comprehensive --scaled 
	python test_accuracy.py --backend numpyro --mode mixed --scaled 
	python test_accuracy.py --backend numpyro --mode generative --scaled 
	python test_speed.py --backend stan --scaled 
	python test_speed.py --backend numpyro --mode comprehensive --scaled 
	python test_speed.py --backend numpyro --mode mixed --scaled 
	python test_speed.py --backend numpyro --mode generative --scaled