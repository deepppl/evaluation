import pandas as pd
import scipy.stats
import os


def preprocess_logs(name, logdir="logs"):
    dfs = []
    duration_files = [f for f in os.listdir(logdir) if f"duration_{name}_" in f]
    n = len(duration_files)
    for i, f in enumerate(duration_files):
        df = pd.read_csv(f"{logdir}/{f}", index_col=0)
        df.columns = [f"{name}_{i}_{x}" for x in df.columns]
        dfs.append(df)

    status_file = next(f for f in os.listdir(logdir) if f"status_{name}_" in f)
    df = pd.read_csv(f"{logdir}/{status_file}", index_col=0)
    df.columns = [f"{name}_{x}" for x in df.columns]
    dfs.append(df)

    df = pd.concat(dfs, axis=1)
    df[f"{name}_time"] = df[[f"{name}_{i}_time" for i in range(n)]].mean(axis=1)
    df[f"{name}_time_std"] = df[[f"{name}_{i}_time" for i in range(n)]].std(axis=1)

    tfrmt = (
        lambda x: x
        if pd.isna(x)
        else f"{x // 3600:02.0f}:{x // 60 % 60:02.0f}:{x % 60:02.0f}"
    )
    df[f"{name}_duration"] = df[f"{name}_time"].apply(tfrmt)

    return df[
        [f"{name}_status", f"{name}_time", f"{name}_time_std", f"{name}_duration"]
    ]


pyro_comprehensive = preprocess_logs("pyro_comprehensive")
numpyro_comprehensive = preprocess_logs("numpyro_comprehensive")
# numpyro_mixed = preprocess_logs("numpyro_mixed")
# numpyro_generative = preprocess_logs("numpyro_generative")
stan = preprocess_logs("stan")

# mean_res = pd.concat([pyro_comprehensive, numpyro_mixed, numpyro_comprehensive, numpyro_generative, stan], axis=1)
mean_res = pd.concat([stan, pyro_comprehensive, numpyro_comprehensive], axis=1)
mean_res["example"] = mean_res.index.map(lambda x: x.split("-")[1])
mean_res["data"] = mean_res.index.map(lambda x: x.split("-")[0])
mean_res = mean_res.sort_values(by="example")
mean_res = mean_res[mean_res.stan_status == "success"]

mean_res["speedup"] = mean_res.stan_time / mean_res.numpyro_comprehensive_time
speedups = mean_res[mean_res.numpyro_comprehensive_status == "success"]["speedup"]
mean_res["speedup"] = speedups

print("\n--- Comparing inference results with PosteriorDB references ---\n")

print(
    mean_res[
        [
            "example",
            "data",
            "stan_status",
            # "stan_time",
            "pyro_comprehensive_status",
            "numpyro_comprehensive_status",
            # "numpyro_comprehensive_time",
            # "numpyro_mixed_status",
            # "numpyro_mixed_time",
            # "numpyro_generative_status",
            # "numpyro_generative_time",
            "speedup",
        ]
    ].to_markdown(index=None)
)


print("\n--- Summary ---\n")

print(f"Total benchs: {len(stan)}")
print(f"Stan successes: {len(stan[stan.stan_status == 'success'])}")
mean_res = mean_res[mean_res.stan_status == "success"]
print(f"Valid speedup: {len(mean_res['speedup'].dropna())}")
print(f"Average speedup: {scipy.stats.gmean(mean_res.speedup.dropna())}")

print("\n------\n")
