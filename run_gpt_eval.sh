export GPT_API_KEY=...
export GPT_API_BASE=...
export MODEL =...          # one of "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"                                                                   
export INFERENCE_TYPE=...  # one of "caption", "narration", "frames", "frames_caption", "frame_ground_truth_narration"
export DATASET_PATH=...
export OUTPUT_DIR=...

clear
python /disks/disk1/share/EgoThink/videos/h2m/egothink/gpt_eval.py \
        --model_name $MODEL \
        --inference_type $INFERENCE_TYPE \
         --api_key $GPT_API_KEY \
        --api_base $GPT_API_BASE \
        --dataset_path $DATASET_PATH \
        --answer_path $OUTPUT_DIR