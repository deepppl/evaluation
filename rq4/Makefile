all: scaled

.PHONY: eval scaled clean cleanall

eval:
	python multimodal.py

scaled: eval

clean:
	rm -rf _tmp multimodal_model multimodal_model.hpp

cleanall: clean
	rm -f deepstan-vs-deepstansvi.pdf deepstansvi-vs-stanadvi.pdf stan-vs-deepstansvi.pdf
