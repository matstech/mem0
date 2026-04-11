#!/usr/bin/env bash

set -euo pipefail

# Run the benchmark pipeline with a fixed mem0_local baseline:
# 1. Run mem0_local add + search + evals once
# 2. Re-run DMF study as needed
# 3. Rebuild the general comparison only when explicitly requested

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_SLUG="${RUN_SLUG:-benchmark_main}"
LOCOMO_JSON="${LOCOMO_JSON:-dataset/locomo10.json}"
LOCOMO_RAG_JSON="${LOCOMO_RAG_JSON:-dataset/locomo10_rag.json}"
RUNTIME_CONFIG_PATH="${RUNTIME_CONFIG_PATH:-config/benchmark_runtime.toml}"
DMF_BASE_CONFIG_PATH="${DMF_BASE_CONFIG_PATH:-config/dmf_benchmark_settings.toml}"
DMF_BASE_BUDGET="${DMF_BASE_BUDGET:-2048}"
PIPELINE_MODE="${PIPELINE_MODE:-all}"
RESET_RUN="${RESET_RUN:-0}"
FORCE_MEM0="${FORCE_MEM0:-0}"
FORCE_DMF_STUDY="${FORCE_DMF_STUDY:-1}"

RESULTS_ROOT="results/${RUN_SLUG}"
RUN_ROOT="run/${RUN_SLUG}"

MEM0_RESULTS_DIR="${RESULTS_ROOT}/mem0_local"
MEM0_RUNTIME_DIR="${RUN_ROOT}/mem0_local"

DMF_STUDY_RESULTS_DIR="${RESULTS_ROOT}/dmf_study"
DMF_STUDY_RUNTIME_DIR="${RUN_ROOT}/dmf_study"

COMPARE_OUTPUT_ROOT="${RESULTS_ROOT}/comparison_summary"
GENERAL_COMPARE_MD="${RESULTS_ROOT}/GENERAL_COMPARE.md"

MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        log "Missing required file: $path"
        exit 1
    fi
}

require_dir() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        log "Missing required directory: $path"
        exit 1
    fi
}

latest_token_accounting_json() {
    local artifact_dir="$1"
    find "$artifact_dir" -maxdepth 2 -type f -name '*token_accounting_*.json' | sort | tail -n 1
}

mem0_outputs_ready() {
    local token_json
    token_json="$(latest_token_accounting_json "$MEM0_RESULTS_DIR" || true)"
    [[ -f "${MEM0_RESULTS_DIR}/mem0_local_results.json" ]] \
        && [[ -f "${MEM0_RESULTS_DIR}/mem0_local_eval_metrics.json" ]] \
        && [[ -n "$token_json" ]] \
        && [[ -f "$token_json" ]]
}

reset_current_run() {
    log "Resetting current run slug: ${RUN_SLUG}"
    rm -rf "$RESULTS_ROOT" "$RUN_ROOT"
}

clean_mem0_outputs() {
    log "Cleaning mem0_local outputs for ${RUN_SLUG}"
    rm -rf "$MEM0_RESULTS_DIR" "$MEM0_RUNTIME_DIR"
    mkdir -p "$MEM0_RESULTS_DIR" "$MEM0_RUNTIME_DIR"
}

clean_dmf_outputs() {
    log "Cleaning DMF study and comparison outputs for ${RUN_SLUG}"
    rm -rf "$DMF_STUDY_RESULTS_DIR" "$DMF_STUDY_RUNTIME_DIR" "$COMPARE_OUTPUT_ROOT"
    rm -f "$GENERAL_COMPARE_MD"
    mkdir -p "$DMF_STUDY_RESULTS_DIR" "$DMF_STUDY_RUNTIME_DIR" "$COMPARE_OUTPUT_ROOT"
}

run_mem0_local() {
    log "Running mem0_local add"
    python3 run_experiments.py \
        --technique_type mem0_local \
        --method add \
        --locomo_json_path "$LOCOMO_JSON" \
        --output_folder "$MEM0_RESULTS_DIR" \
        --runtime_config_path "$RUNTIME_CONFIG_PATH" \
        --mem0_local_output_root "$MEM0_RUNTIME_DIR"

    log "Running mem0_local search"
    python3 run_experiments.py \
        --technique_type mem0_local \
        --method search \
        --locomo_json_path "$LOCOMO_JSON" \
        --output_folder "$MEM0_RESULTS_DIR" \
        --runtime_config_path "$RUNTIME_CONFIG_PATH" \
        --mem0_local_output_root "$MEM0_RUNTIME_DIR"

    require_file "${MEM0_RESULTS_DIR}/mem0_local_results.json"

    log "Running mem0_local evals"
    python3 evals.py \
        --input_file "${MEM0_RESULTS_DIR}/mem0_local_results.json" \
        --output_file "${MEM0_RESULTS_DIR}/mem0_local_eval_metrics.json"

    require_file "${MEM0_RESULTS_DIR}/mem0_local_eval_metrics.json"
}

run_mem0_local_if_needed() {
    if [[ "$FORCE_MEM0" == "1" ]]; then
        clean_mem0_outputs
        run_mem0_local
        return
    fi

    if mem0_outputs_ready; then
        log "mem0_local baseline already available for ${RUN_SLUG}; skipping mem0_local run"
        return
    fi

    clean_mem0_outputs
    run_mem0_local
}

run_dmf_study_phase() {
    local phase="$1"
    shift
    log "Running dmf_study phase=${phase} $*"
    python3 dmf_study/run_study.py \
        --dataset-path "$LOCOMO_RAG_JSON" \
        --runtime-config-path "$RUNTIME_CONFIG_PATH" \
        --base-config-path "$DMF_BASE_CONFIG_PATH" \
        --output-root "$DMF_STUDY_RESULTS_DIR" \
        --runtime-root "$DMF_STUDY_RUNTIME_DIR" \
        --phase "$phase" \
        "$@"
}

run_dmf_study() {
    if [[ "$FORCE_DMF_STUDY" == "1" ]]; then
        clean_dmf_outputs
    else
        mkdir -p "$DMF_STUDY_RESULTS_DIR" "$DMF_STUDY_RUNTIME_DIR" "$COMPARE_OUTPUT_ROOT"
    fi
    run_dmf_study_phase budget
    run_dmf_study_phase active_context --active-step window --base-budget "$DMF_BASE_BUDGET"
    run_dmf_study_phase active_context --active-step pruning --base-budget "$DMF_BASE_BUDGET"
    run_dmf_study_phase ltm --base-budget "$DMF_BASE_BUDGET"
}

run_general_compare() {
    local mem0_token_json
    mem0_token_json="$(latest_token_accounting_json "$MEM0_RESULTS_DIR")"
    require_file "$mem0_token_json"

    local -a compare_args
    compare_args=(
        --comparison-name "${RUN_SLUG}_general_compare"
        --output-root "$COMPARE_OUTPUT_ROOT"
        --notes "General comparison across mem0_local and DMF study variants for ${RUN_SLUG}"
        --baseline "mem0_local,${MEM0_RESULTS_DIR}/mem0_local_results.json,${MEM0_RESULTS_DIR}/mem0_local_eval_metrics.json,${mem0_token_json}"
    )

    local dmf_count=0
    while IFS= read -r results_path; do
        local variant_dir
        local metrics_path
        local token_json
        local label

        variant_dir="$(dirname "$results_path")"
        metrics_path="${variant_dir}/dmf_eval_metrics.json"
        token_json="$(latest_token_accounting_json "$variant_dir")"
        label="$(basename "$variant_dir")"

        require_file "$metrics_path"
        require_file "$token_json"

        compare_args+=(--baseline "${label},${results_path},${metrics_path},${token_json}")
        dmf_count=$((dmf_count + 1))
    done < <(find "$DMF_STUDY_RESULTS_DIR" -mindepth 2 -maxdepth 2 -type f -name 'dmf_results.json' | sort)

    if [[ "$dmf_count" -eq 0 ]]; then
        log "No DMF variant results found under ${DMF_STUDY_RESULTS_DIR}"
        exit 1
    fi

    mkdir -p "$COMPARE_OUTPUT_ROOT"
    find "$COMPARE_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

    log "Running general comparison across mem0_local and ${dmf_count} DMF variants"
    python3 comparison_summary/run_comparison.py "${compare_args[@]}"

    local latest_compare_dir
    latest_compare_dir="$(find "$COMPARE_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${RUN_SLUG}_general_compare_*" | sort | tail -n 1)"
    require_file "${latest_compare_dir}/report.md"

    local report_with_rewritten_paths
    report_with_rewritten_paths="$(python3 - "$latest_compare_dir" "$RESULTS_ROOT" <<'PY'
from pathlib import Path
import sys

artifact_dir = Path(sys.argv[1])
results_root = Path(sys.argv[2])
report_path = artifact_dir / "report.md"
text = report_path.read_text(encoding="utf-8")
relative_artifact_dir = artifact_dir.relative_to(results_root)
text = text.replace("(figures/", f"({relative_artifact_dir.as_posix()}/figures/")
print(text, end="")
PY
)"

    {
        echo "# General Compare"
        echo
        echo "- Run slug: \`${RUN_SLUG}\`"
        echo "- Generated at: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
        echo "- mem0_local results: \`${MEM0_RESULTS_DIR}/mem0_local_results.json\`"
        echo "- mem0_local metrics: \`${MEM0_RESULTS_DIR}/mem0_local_eval_metrics.json\`"
        echo "- dmf_study root: \`${DMF_STUDY_RESULTS_DIR}\`"
        echo "- compared DMF variants: \`${dmf_count}\`"
        echo "- comparison artifact dir: \`${latest_compare_dir}\`"
        echo "- comparison zip: \`${latest_compare_dir}.zip\`"
        echo
        printf '%s\n' "$report_with_rewritten_paths"
    } > "$GENERAL_COMPARE_MD"

    log "General comparison markdown written to ${GENERAL_COMPARE_MD}"
}

main() {
    mkdir -p run results
    if [[ "$RESET_RUN" == "1" ]]; then
        reset_current_run
    fi
    mkdir -p "$MEM0_RESULTS_DIR" "$DMF_STUDY_RESULTS_DIR" "$COMPARE_OUTPUT_ROOT" "$MEM0_RUNTIME_DIR" "$DMF_STUDY_RUNTIME_DIR"

    log "Pipeline run slug: ${RUN_SLUG}"
    log "Pipeline mode: ${PIPELINE_MODE}"
    log "Dataset (mem0_local): ${LOCOMO_JSON}"
    log "Dataset (DMF): ${LOCOMO_RAG_JSON}"
    log "FORCE_MEM0=${FORCE_MEM0} FORCE_DMF_STUDY=${FORCE_DMF_STUDY} RESET_RUN=${RESET_RUN}"

    case "$PIPELINE_MODE" in
        all)
            run_mem0_local_if_needed
            run_dmf_study
            ;;
        mem0)
            run_mem0_local_if_needed
            ;;
        dmf)
            if ! mem0_outputs_ready; then
                log "mem0_local baseline is missing; run PIPELINE_MODE=mem0 or PIPELINE_MODE=all first"
                exit 1
            fi
            run_dmf_study
            ;;
        compare)
            if ! mem0_outputs_ready; then
                log "mem0_local baseline is missing; run PIPELINE_MODE=mem0 or PIPELINE_MODE=all first"
                exit 1
            fi
            require_dir "$DMF_STUDY_RESULTS_DIR"
            run_general_compare
            ;;
        *)
            log "Unsupported PIPELINE_MODE=${PIPELINE_MODE}. Use one of: all, mem0, dmf, compare"
            exit 1
            ;;
    esac

    log "Pipeline completed"
    if [[ "$PIPELINE_MODE" == "compare" ]]; then
        log "Final markdown: ${GENERAL_COMPARE_MD}"
    fi
}

main "$@"
