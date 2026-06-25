"""Combine CONDOR predictions with fresh FlexDC validation results."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from am_condor_flexdc_training_utilities import (
    CONDOR_POWER_COST_COEFFICIENT,
    CONDOR_QOS_BETA,
    CONDOR_QOS_RHO,
    CONDOR_QOS_THRESHOLD,
)


def labels_from_results_csv(path):
    data = pd.read_csv(path)
    if len(data) != 1:
        raise ValueError(f"Expected one FlexDC result row in {path}, found {len(data)}.")
    row = data.iloc[0]
    n = float(row["server_count"])
    j = float(row["workload_mix_size"])

    raw_power = CONDOR_POWER_COST_COEFFICIENT * (
        float(row["P_actual_watts"]) - float(row["R_actual_watts"])
    )
    raw_error = float(row["Mtrack_Error_MeanAbs_Watts"]) / 1000.0
    probabilities = np.asarray(json.loads(row["QoS_Delay_Probabilities"]), dtype=float)
    raw_qos = CONDOR_QOS_BETA * np.logaddexp(
        0,
        CONDOR_QOS_RHO * (probabilities - CONDOR_QOS_THRESHOLD),
    ).sum()
    scaled = np.asarray([raw_power * 120.0 / n, raw_error * 200.0 / n, raw_qos / j])

    return row, np.asarray([raw_power, raw_error, raw_qos]), scaled


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--inference-comparison-csv", required=True)
    parser.add_argument("--start-results-csv", required=True)
    parser.add_argument("--optimized-results-csv", required=True)
    parser.add_argument("--out-csv", default="condor_end_to_end_validation.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    candidate = json.load(open(args.candidate_json))
    cost_weights = np.asarray(candidate["cost_weights"], dtype=float)
    inference = pd.read_csv(args.inference_comparison_csv).set_index("Configuration")

    predicted_rows = {
        "Starting configuration": inference.loc["Starting sampled point"],
        "CONDOR-selected configuration": inference.loc["CONDOR-selected continuous point"],
    }

    rows = []
    for label, path in [
        ("Starting configuration", args.start_results_csv),
        ("CONDOR-selected configuration", args.optimized_results_csv),
    ]:
        source, raw, scaled = labels_from_results_csv(path)
        predicted = predicted_rows[label]
        weight_columns = sorted(
            [name for name in source.index if name.startswith("Weight_") and name != "Weight_Sample_ID"],
            key=lambda name: int(name.split("_")[-1]),
        )

        rows.append({
            "Configuration": label,
            "Pbar_kw_per_server": float(source["Pbar_kw_per_server"]),
            "R_kw_per_server": float(source["R_kw_per_server"]),
            "Weights": json.dumps([float(source[column]) for column in weight_columns]),
            "Predicted_Scaled_cost_power": float(predicted["Predicted_Scaled_cost_power"]),
            "Actual_Scaled_cost_power": scaled[0],
            "Predicted_Scaled_cost_error": float(predicted["Predicted_Scaled_cost_error"]),
            "Actual_Scaled_cost_error": scaled[1],
            "Predicted_Scaled_cost_qos": float(predicted["Predicted_Scaled_cost_qos"]),
            "Actual_Scaled_cost_qos": scaled[2],
            "Predicted_Optimization_Objective": float(predicted["Predicted_Optimization_Objective"]),
            "Actual_Optimization_Objective": float(np.dot(cost_weights, scaled)),
            "Actual_Raw_cost_power": raw[0],
            "Actual_Raw_cost_error": raw[1],
            "Actual_Raw_cost_qos": raw[2],
            "MeanAbs_Normalized_Tracking_Error": float(source["Mtrack_Error_MeanAbs_Normalized"]),
            "QoS_Violation_Ratio": float(source["QoS_Violation_Ratio"]),
        })

    table = pd.DataFrame(rows)
    table.to_csv(args.out_csv, index=False)
    print(table.round(6).to_string(index=False))
    print(f"\nSaved: {Path(args.out_csv)}")


if __name__ == "__main__":
    main()
