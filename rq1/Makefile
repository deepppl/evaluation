all:
	@echo "Help:"
	@echo "  make eval                 # run the full evaluation"
	@echo "  make scaled               # run only a subset of the configuration backend/mode"
	@echo "  make eval_compile         # run the compilation on example-models"
	@echo "  make eval_compile_infer   # run the compilation and inference on posteriordb"
	@echo "  make scaled_compile       # scaled version of eval_compile"
	@echo "  make scaled_compile_infer # scaled version of eval_compile_infer"
	@echo "  make clean                # remove the compiled files"
	@echo "  make cleanall             # remove the compiled files and output files"

.PHONY: all scaled eval eval_compile eval_compile_infer scaled_compile scaled_compile_infer clean cleanall

eval: eval_compile eval_compile_infer

scaled: scaled_compile scaled_compile_infer

eval_compile:
	./test_example-models.sh pyro comprehensive
	./test_example-models.sh pyro mixed
	./test_example-models.sh pyro generative
	./test_example-models.sh numpyro comprehensive
	./test_example-models.sh numpyro mixed
	./test_example-models.sh numpyro generative

eval_compile_infer:
	python test_posteriordb.py --backend pyro --mode comprehensive
	python test_posteriordb.py --backend pyro --mode mixed
	python test_posteriordb.py --backend pyro --mode generative
	python test_posteriordb.py --backend numpyro --mode comprehensive
	python test_posteriordb.py --backend numpyro --mode mixed
	python test_posteriordb.py --backend numpyro --mode generative

scaled_compile:
	./test_example-models.sh pyro comprehensive
	./test_example-models.sh pyro generative

scaled_compile_infer:
	python test_posteriordb.py --backend numpyro --mode comprehensive
	python test_posteriordb.py --backend numpyro --mode generative

clean:
	rm -rf _tmp

cleanall: clean
	rm -rf logs *~
