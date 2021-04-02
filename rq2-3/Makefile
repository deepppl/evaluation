all:
	@echo "Help:"
	@echo "  make eval             # run the full evaluation"
	@echo "  make scaled           # run only a subset of the configuration backend/mode on a subset of examples"
	@echo "  make eval_accuracy    # run the accuracy tests"
	@echo "  make eval_speed       # run the performance tests"
	@echo "  make scaled_accuracy  # scaled version of eval_accuracy"
	@echo "  make scaled_speed     # scaled version of eval_speed"
	@echo "  make clean            # remove the compiled files"
	@echo "  make cleanall         # remove the compiled files and output files"

.PHONY: all eval scaled eval_accuracy eval_speed scaled_accuracy scaled_speed clean cleanall

eval: eval_accuracy eval_speed
	python results_analysis.py

scaled: scaled_accuracy scaled_speed
	python results_analysis.py --nopyro

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
	python test_accuracy.py --backend stan --chains 2 \
		--posterior earnings-earn_height kidiq-kidscore_interaction nes1976-nes eight_schools-eight_schools_noncentered mesquite-mesquite
	python test_accuracy.py --backend numpyro --mode comprehensive --chains 2 \
		--posterior earnings-earn_height kidiq-kidscore_interaction nes1976-nes eight_schools-eight_schools_noncentered mesquite-mesquite
	python test_accuracy.py --backend numpyro --mode mixed --chains 2 \
		--posterior earnings-earn_height kidiq-kidscore_interaction nes1976-nes eight_schools-eight_schools_noncentered mesquite-mesquite
	python test_accuracy.py --backend numpyro --mode generative --chains 2 \
		--posterior earnings-earn_height kidiq-kidscore_interaction nes1976-nes eight_schools-eight_schools_noncentered mesquite-mesquite

scaled_speed:
	python test_speed.py --backend stan --runs 1 --chains 2 \
		--posterior earnings-earn_height kidiq-kidscore_interaction nes1976-nes eight_schools-eight_schools_noncentered
	python test_speed.py --backend numpyro --mode comprehensive --runs 1 --chains 2 \
		--posterior earnings-earn_height kidiq-kidscore_interaction nes1976-nes eight_schools-eight_schools_noncentered
	python test_speed.py --backend numpyro --mode mixed --runs 1 --chains 2 \
		--posterior earnings-earn_height kidiq-kidscore_interaction nes1976-nes eight_schools-eight_schools_noncentered
	python test_speed.py --backend numpyro --mode generative --runs 1 --chains 2 \
		--posterior earnings-earn_height kidiq-kidscore_interaction nes1976-nes eight_schools-eight_schools_noncentered

clean:
	rm -rf _tmp __pycache__ _build_*

cleanall: clean
	rm -rf logs *~
