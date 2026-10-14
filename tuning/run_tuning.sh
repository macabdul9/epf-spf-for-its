#!/usr/bin/env bash
# Run tuning settings for one generator on the 583-item D1K set, budgets 1-128.
#
#   tuning/run_tuning.sh <generator> <setting-id>...     e.g. qwen-omni E1 E2 E3
#   tuning/run_tuning.sh <generator> all
#   tuning/run_tuning.sh <generator> list
#
# Settings run one after another. Each writes its own output file and resumes on a
# re-run. Set DRY_RUN=1 to print the resolved config of each setting without sending
# any request. See tuning/PLAN.md.
set -euo pipefail

usage() {
    echo "usage: $0 <qwen-omni|qwen-omni-3b> <setting-id...|all|list>" >&2
    exit 2
}
[[ $# -ge 2 ]] || usage
generator=$1
shift
case "$generator" in
    qwen-omni | qwen-omni-3b) ;;
    *) usage ;;
esac

# id | needs the 30B judge | epf-spf method flags
SETTINGS=(
    "C1|no|--method epf-project --scorer random"
    "C2|no|--method epf-project --scorer random --resample-from-step 1000"
    "E1|yes|--method epf-paper"
    "E2|yes|--method epf-itshub"
    "E3|yes|--method epf-project"
    "E4|yes|--method epf-project --resample-from-step 2"
    "E5|yes|--method epf-project --resample-from-step 3"
    "E6|yes|--method epf-project --weight-temperature 1"
    "S1|no|--method spf"
    "S2|no|--method spf --gate-threshold 0.7"
    "S3|no|--method spf --resample-rule every_step"
    "S4|no|--method spf --weight-temperature 1"
    "S5|no|--method spf --weight-temperature 1 --gate-threshold 0.7"
    "S6|no|--method spf --weight-temperature 1 --resample-rule every_step"
)

if [[ $1 == list ]]; then
    printf '%s\n' "${SETTINGS[@]}"
    exit 0
fi
if [[ $1 == all ]]; then
    wanted=()
    for entry in "${SETTINGS[@]}"; do wanted+=("${entry%%|*}"); done
else
    wanted=("$@")
fi

: "${EPF_SPF:?set EPF_SPF to the epf-spf executable}"
: "${EPF_DATA_ROOT:?set EPF_DATA_ROOT}"
: "${EPF_EXTENDED_DIR:?set EPF_EXTENDED_DIR}"
: "${GEN_ENDPOINT:?set GEN_ENDPOINT, e.g. http://127.0.0.1:9401/v1}"
: "${OUT_ROOT:?set OUT_ROOT, the folder for result files}"
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
BUDGETS=1,2,4,8,16,32,64,128
BENCH=d1k_csv_orig
export EPF_DATA_ROOT EPF_EXTENDED_DIR

out_dir="$OUT_ROOT/d1k/$generator"
mkdir -p "$out_dir/logs"

for id in "${wanted[@]}"; do
    entry=""
    for candidate in "${SETTINGS[@]}"; do
        [[ ${candidate%%|*} == "$id" ]] && entry=$candidate
    done
    [[ -n $entry ]] || { echo "unknown setting: $id" >&2; exit 2; }
    IFS='|' read -r _ needs_judge flags <<<"$entry"
    read -r -a method_flags <<<"$flags"

    judge_flags=()
    if [[ $needs_judge == yes ]]; then
        : "${JUDGE_ENDPOINT:?setting $id needs JUDGE_ENDPOINT (the 30B judge)}"
        judge_flags=(--judge-endpoint "$JUDGE_ENDPOINT" --judge-model "$JUDGE_MODEL")
    fi
    extra=()
    [[ ${DRY_RUN:-0} == 1 ]] && extra=(--dry-run)

    output="$out_dir/$id.jsonl"
    echo "[$(date '+%F %T')] $generator $id -> $output"
    "$EPF_SPF" "${method_flags[@]}" \
        --bench "$BENCH" --budgets "$BUDGETS" \
        --generator-endpoint "$GEN_ENDPOINT" --generator-model "$generator" \
        "${judge_flags[@]}" \
        --output "$output" "${extra[@]}" \
        2>&1 | tee -a "$out_dir/logs/$id.log"
done
