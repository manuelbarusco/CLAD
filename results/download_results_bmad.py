import wandb
import json
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm

credentials_path = "/home/manuel_barusco/continual_learning/CL_VAD/adcl_paper/configurations/credentials.json"
models = ["patchcore"]
metrics = [
    "Summary/evaluation_ad/f1",
    "Summary/evaluation_ad/f1_img",
    "Summary/evaluation_ad/img_roc_auc",
    "Summary/evaluation_ad/per_pixel_rocauc",
    "Summary/evaluation_ad/au_pro_score",
    "Summary/evaluation_ad/pr_auc_score"
]


strategies = ["multi_task", "single_model", "cl-10000", "cl-30000"]

def return_summaries_dataframes(api: wandb.Api, credentials: json) -> pd.DataFrame:

    #define the dataframe columns
    data = dict()
    data["Model"] = list()
    data["Seed"] = list()
    data["Strategy"] = list()
    data["Memory Size"] = list()
    for metric in metrics:
        data[metric.split("/")[-1]] = list()
    data["Relative Gap"] = list()
    data["Avg Forgetting"] = list()
    data["Runtime"] = list()

    for model in models:
        print(model)

        last_task_perf_multi = None

        for strategy in strategies:
            runs = api.runs(path = f"{credentials['entity']}/{credentials['project_name']}", filters = {"$and": [{"tags": model}, {"tags": strategy},  {"state": "finished"}]})

            for run in tqdm(runs):

                data["Model"].append(model)

                if strategy in ["cl-10000", "cl-30000"]:
                    data["Strategy"].append("replay")
                    data["Memory Size"].append(run.config["hyperparameters"]["mem_size"])
                else:
                    data["Strategy"].append(strategy)
                    data["Memory Size"].append(0)

                runtime = time.strftime("%H:%M:%S", time.gmtime(run.summary["_runtime"]))

                data["Runtime"].append(runtime)

                if "seed" in run.config["hyperparameters"].keys():
                    data["Seed"].append(run.config["hyperparameters"]["seed"])
                else:
                    data["Seed"].append("43")

                for metric in metrics:
                    #print(metric)

                    history = run.history(keys=[metric])

                    data[metric.split("/")[-1]].append(round(history[metric].values[-1],2))

                    if strategy == "multi_task" and metric == "Summary/evaluation_ad/f1_img":
                        last_task_perf_multi = round(history[metric].values[-1],2)
                        print(last_task_perf_multi)

                    if metric == "Summary/evaluation_ad/f1_img":
                        if (strategy in ["cl-10000", "cl-30000", "naive"] ) and last_task_perf_multi:
                            print(history[metric].values[-1],2)
                            data["Relative Gap"].append(round(last_task_perf_multi - history[metric].values[-1],2) * 100)
                        else:
                            data["Relative Gap"].append(0)

                        if strategy in ["cl-10000", "cl-30000", "naive"]:
                            data["Avg Forgetting"].append(calculate_avg_forgetting(run))
                        else:
                            data["Avg Forgetting"].append(0)

    for key, values in data.items():
        print(f"{key}: {len(values)}")

    return pd.DataFrame(data)


def calculate_avg_forgetting(run):
    avg_forgetting = 0

    last_f1_values = run.history(keys=[f"Task_Results/T5/evaluation_ad/f1_img"])[f"Task_Results/T5/evaluation_ad/f1_img"].values

    best_f1_values = list()

    for i in range(0,6):
        best_f1_values.append(run.history(keys=[f"Task_Results/T{i}/evaluation_ad/f1_img"])[f"Task_Results/T{i}/evaluation_ad/f1_img"].values[i])

    for i in range(0,6):
        avg_forgetting += (best_f1_values[i] - last_f1_values[i]) / best_f1_values[i]

        if best_f1_values[i] - last_f1_values[i] < 0:
            print(f"Task {i}, Run {run.name}")

    return (avg_forgetting / 9) * 100


def avg_runs(cl_runs: dict, data: pd.DataFrame) -> pd.DataFrame:

    for scenario, rep in cl_runs.items():
        columns = {}

        for i in range(10):
            columns[i] = list()

        for i in range(len(rep)):
            for task_index in range(10):
                columns[task_index].append(rep[i][task_index])

        avg_df = pd.DataFrame(data = columns)

        avg_df = avg_df.mean()

        data[f"replay-{scenario}"] = avg_df.to_list()


def get_plot_dfs(api: wandb.Api, credentials: json, seed: int):

    for model in tqdm(models):
        #get all the runs for that model

        data = dict()
        data["task_index"] = list(i for i in range(6))

        for strategy in strategies:
            runs = api.runs(path = f"{credentials['entity']}/{credentials['project_name']}", filters = {"$and": [{"tags": model}, {"tags": strategy}, {"state": "finished"}]})

            for run in runs:
                if strategy == "multi_task":
                    data[strategy] = [run.history(keys=["Summary/evaluation_ad/f1_img"])["Summary/evaluation_ad/f1_img"].iloc[-1]] * 6
                else:
                    data[strategy] = run.history(keys=["Summary/evaluation_ad/f1_img"])["Summary/evaluation_ad/f1_img"]

        print(data)
        data = pd.DataFrame(data)
        data.to_csv(f'csv/{model}_bmad.csv', index=False)

def main():
    #get the wandb credentials
    f = open(f"{credentials_path}", "rb")
    credentials = json.load(f)
    f.close()

    #login to the wandb API
    api = wandb.Api(api_key = credentials["api_token"])

    table_df = return_summaries_dataframes(api, credentials)

    table_df.to_csv(f'csv/patchcore_final_table_bmad.csv', index=False)

    get_plot_dfs(api, credentials, 43)


if __name__ == "__main__" :
    main()
