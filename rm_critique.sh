# python3 gpt_eval.py --model_name "gpt-4o-mini" --dataset_path "/apdcephfs_cq10/share_1150325/csj/videgothink/final_goalstep_rm_critique.json" --answer_path "./answer/rm_critique" --task "rm_critique"
python3 gpt_eval.py --model_name "gpt-4o" --dataset_path "/apdcephfs_cq10/share_1150325/csj/videgothink/final_rm_critique.json" --answer_path "./answer/rm_critique" --task "rm_critique"
python3 gpt_eval.py --model_name "gpt-4o" --dataset_path "/apdcephfs_cq10/share_1150325/csj/videgothink/test_rm_feedback_v1.json" --answer_path "./answer/rm_feedback" --task "rm_feedback"

