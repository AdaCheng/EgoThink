"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import pandas as pd
model_order = [
    'gpt4v',
    'openflamingo',
    'blip2-7b',
    'otter-image',
    'pandagpt-7b',
    'instructblip-7b',
    'llama-adapter-v2',
    'minigpt-4',
    'mplug-owl',
    'llava-7b',
    'pandagpt-13b',
    'instructblip-13b-vicuna',
    'blip2-13b',
    'llava-13b',
    'instructblip-13b-flant5',
    'llava-1.5-7b',
    'llava-1.5-13b',
    'llava-13b-llama2',
    'sharegpt4v',
    'gpt-4o-frames',
    'qwen2_vl',
    'qwen-vl-chat',
    'gpt-4o',
]

def display_result_single(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    # import pdb; pdb.set_trace()
    # df_all = df_all.sort_values(by=['model', 'question_id', 'tstamp'], ascending=False)
    # df_all = df_all.drop_duplicates(subset=['model', 'question_id'], keep='first')
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]
    
    for model in df["model"].unique():
        # print(f"\n########## {model} ##########")
        df_model = df[df["model"] == model]
        print(f"Number of instances for {model}: {len(df_model)}")
        # df_model = df_model.groupby(["turn"]).mean()
        # print(df_model.sort_values(by="score", ascending=False))
        # print(df[])

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    # print(df_1.sort_values(by="score", ascending=False))
    
    df_1_reset = df_1.reset_index()
    # import pdb; pdb.set_trace()

    order = [x for x in model_order if x in df_1_reset['model'].tolist()]
    order += [x for x in df_1_reset['model'].tolist() if x not in model]
    # print(order)
    df_1_ordered = df_1_reset.set_index('model').loc[order]
    print(df_1_ordered)




    # if args.bench_name == "mt_bench":
    #     print("\n########## Second turn ##########")
    #     df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
    #     print(df_2.sort_values(by="score", ascending=False))

    #     print("\n########## Average ##########")
    #     df_3 = df[["model", "score"]].groupby(["model"]).mean()
    #     print(df_3.sort_values(by="score", ascending=False))


def display_result_pairwise(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df_all = df_all[(df_all["g1_winner"] != "error") & (df_all["g2_winner"] != "error")]

    model_list = (
        df_all["model_1"].unique().tolist() + df_all["model_2"].unique().tolist()
    )
    model_list = list(set(model_list))

    list_res = []
    # traverse df row by row
    for index, row in df_all.iterrows():
        if args.model_list is not None and row["model_1"] not in args.model_list:
            continue
        if args.baseline_model is not None:
            if args.baseline_model not in [row["model_1"], row["model_2"]]:
                continue
        if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"]:
            list_res.append({"model": row["model_1"], "win": 0, "loss": 0, "tie": 1})
            list_res.append({"model": row["model_2"], "win": 0, "loss": 0, "tie": 1})
        else:
            if row["g1_winner"] == "model_1":
                winner = row["model_1"]
                loser = row["model_2"]
            else:
                winner = row["model_2"]
                loser = row["model_1"]
            list_res.append({"model": winner, "win": 1, "loss": 0, "tie": 0})
            list_res.append({"model": loser, "win": 0, "loss": 1, "tie": 0})

    df = pd.DataFrame(list_res)
    df = df.groupby(["model"]).sum()

    # remove baseline model
    if args.baseline_model is not None:
        df = df[df.index != args.baseline_model]
    # add win rate
    df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])
    df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
    # each tie counts as 0.5 win + 0.5 loss
    df["win_rate_adjusted"] = (df["win"] + 0.5 * df["tie"]) / (
        df["win"] + df["loss"] + df["tie"]
    )
    # print(df.sort_values(by="win_rate", ascending=False))
    # print(df.sort_values(by="loss_rate", ascending=True))
    print(df.sort_values(by="win_rate_adjusted", ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
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
    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise
    if args.input_file is None:
        args.input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_{args.mode}.jsonl"
        )

    print(f"Mode: {args.mode}")
    display_result_func(args)
