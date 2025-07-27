#################################################################################
# Globals                                                                       #
#################################################################################

PYTHON_INTERPRETER = uv run python


#################################################################################
# Project Commands                                                              #
#################################################################################

## formatting and lint
.PHONY: lint
LINT_TARGET=src
lint:
	uv run ruff format $(LINT_TARGET)
	uv run ruff check --fix $(LINT_TARGET)


## setup
.PHONY: setup
setup: environment dataset


## make environment
.PHONY: environment
environment:
	uv sync 


## make dataset
.PHONY: dataset
dataset: datasets/livedoor/all.jsonl

data/text/README.txt:
	bash src/download.sh

datasets/livedoor/all.jsonl: data/text/README.txt
	@$(PYTHON_INTERPRETER) src/prepare.py


## login huggingface
.PHONY: login_huggingface
login_huggingface:
	uv run hf auth login


## run mlflow ui
.PHONY: mlflow_ui
mlflow_ui:
	uv run mlflow ui -h 0.0.0.0


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
