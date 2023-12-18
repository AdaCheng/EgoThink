"""
Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import datetime, pytz 
import pandas as pd
from datasets import load_dataset

import numpy as np
from tqdm import tqdm

from common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)


def make_match(
    questions,
    models,
    model_answers,
    judge,
    baseline_model,
    ref_answers=None,
    multi_turn=False,
    ref_model_name='ground_truth'
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m_1 = models[i]
            m_2 = baseline_model
            if m_1 == m_2:
                continue
            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[baseline_model][q_id]
            if ref_answers is not None:
                ref = ref_answers[ref_model_name][q_id]
                match = MatchPair(
                    dict(q),
                    m_1,
                    m_2,
                    a_1,
                    a_2,
                    judge,
                    ref_answer=ref,
                    multi_turn=multi_turn,
                )
            else:
                match = MatchPair(
                    dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                )
            matches.append(match)
    return matches


def make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
    ref_model_name='ground_truth'
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                q_id = q["question_id"]
                m_1 = models[i]
                m_2 = models[j]
                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[m_2][q_id]
                if ref_answers is not None:
                    ref = ref_answers[ref_model_name][q_id]
                    match = MatchPair(
                        dict(q),
                        m_1,
                        m_2,
                        a_1,
                        a_2,
                        judge,
                        ref_answer=ref,
                        multi_turn=multi_turn,
                    )
                else:
                    match = MatchPair(
                        dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                    )
                matches.append(match)
    return matches


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
    ref_model_name='ground_truth'
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[ref_model_name][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def refresh_ref(bench_name):
    primary, secondary = bench_name.split('/')
    dataset = load_dataset("vleai_bench", primary, download_mode='force_redownload', ignore_verifications=True)[secondary]
    ref_answers = []
    question_files = []
    for q_id in range(len(dataset)):
        question = dataset[q_id]['question']
        answer = dataset[q_id]['answer']
        ref_answers.append({
            'question_id': q_id,
            'model_id': 'ground_truth',
            'choices':[{'index': 0, "turns": [answer]}]
        })
        question_files.append({
            'question_id': q_id,
            'turns': [question]
        })
    with open(f"data/{args.bench_name}/question.jsonl", 'w') as f:
        for q in question_files:
            f.write(json.dumps(q) + '\n')
    with open(f"data/{args.bench_name}/reference_answer/ground_truth.jsonl", 'w') as f:
        for ref in ref_answers:
            f.write(json.dumps(ref) + '\n')
    


def make_judge_pairwise(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    # judges["default-mt"] = Judge(
    #     judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True
    # )
    # judges["math-mt"] = Judge(
    #     judge_model,
    #     judge_prompts["pair-math-v1-multi-turn"],
    #     ref_based=True,
    #     multi_turn=True,
    # )
    return judges


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v2"], ref_based=True)
    return judges

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder",
        type=str,
        default="data",
        help="The folder to store the data.",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        '--refresh-ref',
        action='store_true',
        help='Refresh reference answers.'
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument('--ref-model', type=str, default='ground_truth')
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
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

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = (
            f"{data_folder}/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = (
            f"{data_folder}/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
        if args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

    if os.path.exists(output_file):
        # action = input(f"File {output_file} exists. Overwrite? (y/n)")
        if not args.overwrite:
            beijing_time = pytz.timezone('Asia/Shanghai')
            now = datetime.datetime.now(beijing_time).strftime("%m%d%H%M%S")
            output_file = output_file.replace('.jsonl', f'_{now}.jsonl')
        else:
            os.remove(output_file)
    # check_data(questions, model_answers, ref_answers, models, judges)

    # question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    # question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    # matches += make_match_func(
    #     questions, models, model_answers, judges["default"], baseline_model
    # )
    matches += make_match_func(
        questions,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
        ref_model_name=args.ref_model
    )

    match_stat = {}
    match_stat["bench_name"] = args.bench_name
    match_stat["mode"] = args.mode
    match_stat["judge"] = args.judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    # input("Press Enter to confirm...")

    # Play matches
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass

    # Finally sort the output file according to model and question id
    df = pd.read_json(output_file, lines=True)
    df = df.sort_values(by=["model", "question_id"])
    df.to_json(output_file, orient="records", lines=True)
    
