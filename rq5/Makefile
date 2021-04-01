all:
	@echo "Help:"
	@echo "  make eval             # run the full evaluation"
	@echo "  make scaled           # run only one epoch"
	@echo "  make eval_vae         # run the VAE example"
	@echo "  make eval_mlp         # run the MLP example"
	@echo "  make scaled_vae       # run one epoch of the VAE example"
	@echo "  make scaled_mlp       # run one epoch of the MLP example"
	@echo "  make clean            # remove the compiled files"
	@echo "  make cleanall         # remove the compiled files and output files"

.PHONY: all eval scaled eval_vae eval_mlp scaled_vae scaled_mlp clean cleanall

eval: eval_vae eval_mlp

scaled: scaled_vae scaled_mlp

eval_vae:
	python vae.py

eval_mlp:
	python mlp.py

scaled_vae:
	python vae.py -epochs 1

scaled_mlp:
	python mlp.py -epochs 1

clean:
	rm -rf _tmp

cleanall: clean
	rm -rf data DeepStanSVI-vae.csv Pyro-vae.csv pyro-vs-deepstan.pdf
