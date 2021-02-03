import pandas as pd
import scipy.stats


def preprocess(name, csvfile):
    df = pd.read_csv(csvfile, index_col=0)
    tfrmt = (
        lambda x: x
        if pd.isna(x)
        else f"{x // 3600:02.0f}:{x // 60 % 60:02.0f}:{x % 60:02.0f}"
    )
    df["duration"] = df["time"].apply(tfrmt)
    df.columns = [f"{name}_{x}" for x in df.columns]
    return df


pyro_mixed = preprocess("pyro_mixed", "201104_1633_pyro_mixed.csv")
numpyro_mixed = preprocess("numpyro_mixed", "201119_1324_numpyro_mixed.csv")
numpyro_comprehensive = preprocess(
    "numpyro_comprehensive", "201119_1432_numpyro_comprehensive.csv"
)
numpyro_generative = preprocess(
    "numpyro_generative", "201119_1456_numpyro_generative.csv"
)
stan = preprocess("stan", "201114_0927_stan.csv")

results = pd.concat(
    [pyro_mixed, numpyro_mixed, numpyro_comprehensive, numpyro_generative, stan], axis=1
)
results = results[results.stan_status != "mismatch"]
results["example"] = results.index.map(lambda x: x.split("-")[1])
results["data"] = results.index.map(lambda x: x.split("-")[0])
results = results.sort_values(by="example")
results["speedup"] = results.stan_time / results.numpyro_mixed_time

print(f"average speedup {scipy.stats.gmean(results.speedup.dropna())}")

print(
    results[
        [
            "example",
            "data",
            "stan_status",
            "stan_duration",
            "pyro_mixed_status",
            "pyro_mixed_duration",
            "numpyro_mixed_status",
            "numpyro_mixed_duration",
            "numpyro_comprehensive_status",
            "numpyro_comprehensive_duration",
            "numpyro_generative_status",
            "numpyro_generative_duration",
            "speedup",
        ]
    ].to_markdown(index=False)
)
