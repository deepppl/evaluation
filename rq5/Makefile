eval: vae_model.stan vae.py mlp_model.stan mlp.py
	python vae.py
	python mlp.py

scaled: vae_model.stan vae.py mlp_model.stan mlp.py
	python vae.py --epochs 1
	python mlp.py --epochs 1
