lint:
	ruff check .

tests:
	pytest ./test

type:
	pyright validator

qa:
	make lint
	pip install ".[dev]"
	make type
	make tests