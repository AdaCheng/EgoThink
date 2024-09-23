python3 gen_judgment.py \
 --data-folder "./answer" \
 --bench-name "rm_feedback" \
 --mode single \
 --model-list "gpt-4o-frames" \
 --judge-model "gpt-4o" \
 --parallel 4 \
 --judge-file judge_prompts.jsonl