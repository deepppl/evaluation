all: scaled

.PHONY: eval scaled eval_vae eval_mlp scaled_vae scaled_mlp clean cleanall

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
