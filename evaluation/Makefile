
PROGRESS_MODE ?= detailed

ifeq ($(LOG_PROGRESS),1)
PROGRESS_ARGS := --log_progress --progress_mode $(PROGRESS_MODE)
else ifeq ($(LOG_PROGRESS),true)
PROGRESS_ARGS := --log_progress --progress_mode $(PROGRESS_MODE)
else
PROGRESS_ARGS :=
endif

# Run the experiments
run-mem0-add:
	python run_experiments.py --technique_type mem0 --method add $(PROGRESS_ARGS)

run-mem0-search:
	python run_experiments.py --technique_type mem0 --method search --output_folder results/ --top_k 30 $(PROGRESS_ARGS)

run-mem0-plus-add:
	python run_experiments.py --technique_type mem0 --method add --is_graph $(PROGRESS_ARGS)

run-mem0-plus-search:
	python run_experiments.py --technique_type mem0 --method search --is_graph --output_folder results/ --top_k 30 $(PROGRESS_ARGS)

run-mem0-local-add:
	python run_experiments.py --technique_type mem0_local --method add --output_folder results/ $(PROGRESS_ARGS)

run-mem0-local-add-resume:
	python run_experiments.py --technique_type mem0_local --method add --output_folder results/ --resume $(PROGRESS_ARGS)

run-mem0-local-search:
	python run_experiments.py --technique_type mem0_local --method search --output_folder results/ --top_k 30 $(PROGRESS_ARGS)

run-mem0-local: run-mem0-local-add run-mem0-local-search

run-mem0-local-search-resume:
	python run_experiments.py --technique_type mem0_local --method search --output_folder results/ --top_k 30 --resume $(PROGRESS_ARGS)

run-mem0-local-resume: run-mem0-local-add-resume run-mem0-local-search-resume

run-rag:
	python run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 1 --output_folder results/

run-full-context:
	python run_experiments.py --technique_type rag --chunk_size -1 --num_chunks 1 --output_folder results/

run-langmem:
	python run_experiments.py --technique_type langmem --output_folder results/

run-zep-add:
	python run_experiments.py --technique_type zep --method add --output_folder results/

run-zep-search:
	python run_experiments.py --technique_type zep --method search --output_folder results/

run-openai:
	python run_experiments.py --technique_type openai --output_folder results/

run-dmf:
	python run_experiments.py --technique_type dmf --output_folder results/

# Comparison Summary
# Usage: make compare NAME=my_run BASELINES="Label1,res1,met1 Label2,res2,met2" NOTES="Optional notes"
NAME ?= comparison_$(shell date +%Y%m%d_%H%M%S)
NOTES ?= "Generated via Makefile"
BASELINES ?= ""

compare:
	@if [ -z "$(BASELINES)" ]; then \
		echo "Error: BASELINES variable is empty. Usage: make compare BASELINES=\"Label,res_path,met_path ...\""; \
		exit 1; \
	fi; \
	CMD="python3 comparison_summary/run_comparison.py --comparison-name \"$(NAME)\" --notes \"$(NOTES)\""; \
	for b in $(BASELINES); do \
		CMD="$$CMD --baseline $$b"; \
	done; \
	eval $$CMD

clean-comparisons:
	rm -rf results/comparison_summary/*
