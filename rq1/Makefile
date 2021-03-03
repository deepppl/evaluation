all: scaled

.PHONY: scaled eval eval_compile eval_compile_infer scaled_compile_infer clean cleanall

eval: eval_compile eval_compile_infer

scaled: eval_compile scaled_compile_infer

eval_compile:
	./test_example-models.sh

eval_compile_infer:
	python test_posteriordb.py  --backend pyro --mode comprehensive
	python test_posteriordb.py  --backend pyro --mode mixed
	python test_posteriordb.py  --backend pyro --mode generative
	python test_posteriordb.py  --backend numpyro --mode comprehensive
	python test_posteriordb.py  --backend numpyro --mode mixed
	python test_posteriordb.py  --backend numpyro --mode generative

scaled_compile_infer:
	python test_posteriordb.py  --backend numpyro --mode comprehensive
	python test_posteriordb.py  --backend numpyro --mode generative

clean:
	rm -rf _tmp

cleanall: clean
	rm -rf logs *~
