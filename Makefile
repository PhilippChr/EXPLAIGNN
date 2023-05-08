EXPLAIGNN_ROOT := $(shell pwd)

style:
	black --line-length 100 --target-version py38 .

install_clocq:
	git clone https://github.com/PhilippChr/CLOCQ.git && \
	cd CLOCQ/ && \
	pip install -e . && \
	cd $(EXPLAIGNN_ROOT)