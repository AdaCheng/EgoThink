import json
import argparse
from typing import Optional
import glob
import os

def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder",
        type=str,
        default="./answer/",
        help="The folder to store the data.",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="rm_critique",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        '--refresh-ref',
        action='store_true',
        help='Refresh reference answers.'
    )
    parser.add_argument('--ref-model', type=str, default='ground_truth')
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output file.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    data_folder = args.data_folder
    question_file = f"{data_folder}/{args.bench_name}/question.jsonl"
    answer_dir = f"{data_folder}/{args.bench_name}/model_answer"
    ref_answer_dir = f"{data_folder}/{args.bench_name}/reference_answer"

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    # import pdb ; pdb.set_trace()
    ref_answers = load_model_answers(ref_answer_dir)

    scores = {}
    for model, answers in model_answers.items():
        scores[model] = []
        for qid, ans in answers.items():
            ref_ans = ref_answers[args.ref_model][qid]['choices'][0]['turns'][0].strip(".").lower()
            model_ans = ans['choices'][0]['turns'][0].strip(".").lower()
            if ref_ans == model_ans:
                scores[model].append(1)
            else:
                scores[model].append(0)
    
    results = {}
    for model, score in scores.items():
        results[model] = sum(score) / len(score)

    print(results)

    output_file = "/apdcephfs_cq10/share_1150325/csj/EgoThink/results/rm_critique.json"
    with open(output_file, "w") as fout:
        json.dump(results, fout, indent=4)

    # check_data(questions, model_answers, ref_answers, models, judges)
