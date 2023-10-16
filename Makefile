.PHONY: build
build:
	sphinx-build -b html docs/source/ docs/build/html
	python -m build
	twine check dist/*

.PHONY: buildt
buildt:
	python -m pipenv install --dev -e .

.PHONY: clean
clean:
	rm -rf docs/build/html
	rm -rf dist
	rm -rf src/pjimg.egg-info
	rm -rf src/pjimg/__pycache__
	rm -rf src/pjimg/*.pyc
	rm -rf examples/__pycache__
	rm -rf tests/__pycache__
	rm -rf thurible/__pycache__
	rm -f *.log
	python -m pipenv uninstall pjimg
	python -m pipenv install --dev -e .

.PHONY: docs
docs:
# 	python examples/blends/build_doc_images.py
# 	python examples/eases/build_doc_images.py
# 	python examples/filters/build_doc_images.py
# 	python examples/sources/build_doc_images.py
	sphinx-build -b html docs/source/ docs/build/html

.PHONY: pre
pre:
	python precommit.py
	git status

.PHONY: test
test:
	python -m pytest tests/test_util --capture=fd
	python -m pytest tests/test_imgio --capture=fd
	python -m pytest tests/test_sources --capture=fd
	python -m pytest tests/test_filters --capture=fd
	python -m pytest tests/test_eases --capture=fd
	python -m pytest tests/test_blends --capture=fd

.PHONY: testv
testv:
	python -m pytest -vv --capture=fd
