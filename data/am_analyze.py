import re
import numpy as np
import pandas as pd

INPUT_CSV = "all_data.csv"
WEIGHTS_OUT = "condor_weights_only.csv"

# Set this to a specific size like 8 or 16 if you only want per-position averages for that size.
# Leave as None to print per-position averages for all workload sizes.
WORKLOAD_SIZE_TO_INSPECT = None

df = pd.read_csv(INPUT_CSV)


def parse_workload_mix(s):
    """
    workload_mix format:
    [
      [min_power, max_power, min_runtime, max_runtime, qos_threshold, job_size, weight],
      ...
    ]

    Weight is column 6 of each job row.
    """
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(s))
    values = np.array([float(x) for x in nums])

    if len(values) % 7 != 0:
        raise ValueError(f"Cannot reshape workload_mix with {len(values)} values")

    J = len(values) // 7
    return values.reshape(J, 7)


def normalized_entropy(weights):
    """
    1.0 = perfectly equal weights.
    Lower = more skewed.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights[weights > 0]

    if len(weights) <= 1:
        return 0.0

    entropy = -np.sum(weights * np.log(weights))
    return float(entropy / np.log(len(weights)))


def effective_num_jobs(weights):
    """
    Equal weights over J jobs gives J.
    More skewed weights give a smaller number.
    """
    weights = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(weights ** 2))


weight_rows = []
summary_rows = []

for row_idx, row in df.iterrows():
    mix = parse_workload_mix(row["workload_mix"])
    weights = mix[:, 6].astype(float)

    J = len(weights)
    equal_weight = 1.0 / J

    # Uncomment to inspect row-by-row parsing.
    # print(f"\nRow {row_idx}")
    # print("J:", J)
    # print("weights:", weights)
    # print("sum:", weights.sum())

    for job_idx, weight in enumerate(weights):
        weight_rows.append({
            "row_index": row_idx,
            "workload_mix_size": J,
            "job_index": job_idx,
            "weight": weight,
            "equal_weight": equal_weight,
            "weight_minus_equal": weight - equal_weight,
            "weight_over_equal": weight / equal_weight,
        })

    summary_rows.append({
        "row_index": row_idx,
        "workload_mix_size": J,
        "weight_sum": weights.sum(),
        "min_weight": weights.min(),
        "max_weight": weights.max(),
        "mean_weight": weights.mean(),
        "std_weight": weights.std(),
        "range_weight": weights.max() - weights.min(),
        "max_over_equal": weights.max() / equal_weight,
        "min_over_equal": weights.min() / equal_weight,
        "normalized_entropy": normalized_entropy(weights),
        "effective_num_jobs": effective_num_jobs(weights),
        "is_equal_weight_row": np.allclose(weights, np.ones(J) / J, rtol=1e-6, atol=1e-8),
    })


weights_df = pd.DataFrame(weight_rows)
summary_df = pd.DataFrame(summary_rows)

weights_df.to_csv(WEIGHTS_OUT, index=False)

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 180)

print("\nParsed rows:", len(summary_df))
print("Saved weights to:", WEIGHTS_OUT)

print("\nRows by workload size:")
print(summary_df["workload_mix_size"].value_counts().sort_index())

print("\nEqual-weight rows by workload size:")
print(summary_df.groupby("workload_mix_size")["is_equal_weight_row"].sum())


print("\nOverall individual-weight distribution by workload size:")
individual_weight_distribution = (
    weights_df
    .groupby("workload_mix_size")
    .agg(
        count=("weight", "count"),
        mean_weight=("weight", "mean"),
        median_weight=("weight", "median"),
        min_weight=("weight", "min"),
        max_weight=("weight", "max"),
        std_weight=("weight", "std"),
    )
    .reset_index()
)

print(individual_weight_distribution)


print("\nRow-level skew summary by workload size:")
skew_summary = (
    summary_df
    .groupby("workload_mix_size")
    .agg(
        rows=("row_index", "count"),
        equal_rows=("is_equal_weight_row", "sum"),

        median_max_weight=("max_weight", "median"),
        p90_max_weight=("max_weight", lambda x: x.quantile(0.90)),
        max_seen=("max_weight", "max"),

        median_min_weight=("min_weight", "median"),
        p10_min_weight=("min_weight", lambda x: x.quantile(0.10)),
        min_seen=("min_weight", "min"),

        median_max_over_equal=("max_over_equal", "median"),
        p90_max_over_equal=("max_over_equal", lambda x: x.quantile(0.90)),

        median_effective_num_jobs=("effective_num_jobs", "median"),
        mean_effective_num_jobs=("effective_num_jobs", "mean"),

        median_normalized_entropy=("normalized_entropy", "median"),
        mean_normalized_entropy=("normalized_entropy", "mean"),
    )
    .reset_index()
)

print(skew_summary)


print("\nPer-weight-position average by workload size:")
position_average = (
    weights_df
    .groupby(["workload_mix_size", "job_index"])
    .agg(
        count=("weight", "count"),
        mean_weight=("weight", "mean"),
        median_weight=("weight", "median"),
        min_weight=("weight", "min"),
        max_weight=("weight", "max"),
        mean_over_equal=("weight_over_equal", "mean"),
    )
    .reset_index()
)

if WORKLOAD_SIZE_TO_INSPECT is None:
    print(position_average)
else:
    print(position_average[position_average["workload_mix_size"] == WORKLOAD_SIZE_TO_INSPECT])


# Uncomment to inspect extracted weights directly.
# print("\nweights_df head:")
# print(weights_df.head(30))

# Uncomment to inspect row-level summaries directly.
# print("\nsummary_df head:")
# print(summary_df.head(20))

print("\nInterpretation guide:")
print("- mean_weight will always be close to 1/J because each row sums to 1.")
print("- Use median_max_weight, p90_max_weight, max_over_equal, entropy, and effective_num_jobs to see skew.")
print("- normalized_entropy close to 1 means balanced; lower means more skew.")
print("- effective_num_jobs close to J means balanced; closer to 1 means one job dominates.")