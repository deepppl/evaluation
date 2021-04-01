all:
	@echo "Help:"
	@echo "  make eval       # run the full evaluation"
	@echo "  make scaled     # same as eval"
	@echo "  make clean      # remove the compiled files"
	@echo "  make cleanall   # remove the compiled files and output files"

.PHONY: all eval scaled clean cleanall

eval:
	python multimodal.py

scaled: eval

clean:
	rm -rf _tmp multimodal_model multimodal_model.hpp

cleanall: clean
	rm -f deepstan-vs-deepstansvi.pdf deepstansvi-vs-stanadvi.pdf stan-vs-deepstansvi.pdf
