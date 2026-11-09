# Uses whatever python is active (conda env or venv).
# Override with:  make check PY=/path/to/python
PY ?= python
.PHONY: check test tokenizer bins smoke clean

check:            ## Phase-1 gate: params, shapes, forward/backward
	$(PY) -m scripts.check_model

test: check       ## everything that must pass before renting a GPU

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +

tokenizer:        ## Step 2: train 8192 BPE (one-time, irreversible)
	$(PY) -m data_prep.train_tokenizer --target-mb 300

bins:             ## Step 3: text -> uint16 shards (val first, then A, then B)
	$(PY) -m data_prep.build_bins --arm all

smoke:            ## Step 4 gate: 4 optimizer steps + 2 evals on this machine
	$(PY) train.py --model 8M --arm A --smoke --steps 4 --eval-tokens 131072 --run-id smoke8M
