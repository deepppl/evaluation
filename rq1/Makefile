eval: test_example-models.sh test_posteriordb.py compile.sh
	./test_example-models.sh 
	python test_posteriordb.py  --backend pyro --mode comprehensive 
	python test_posteriordb.py  --backend pyro --mode mixed 
	python test_posteriordb.py  --backend pyro --mode generative 
	python test_posteriordb.py  --backend numpyro --mode comprehensive 
	python test_posteriordb.py  --backend numpyro --mode mixed 
	python test_posteriordb.py  --backend numpyro --mode generative

scaled: test_example-models.sh test_posteriordb.py compile.sh
	./test_example-models.sh 
	python test_posteriordb.py  --backend numpyro --mode comprehensive 
	python test_posteriordb.py  --backend numpyro --mode generative
