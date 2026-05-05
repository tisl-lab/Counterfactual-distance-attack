from pathlib import Path
import pandas as pd
import numpy as np

rows = []
# root = Path("dataset")  # adjust if your datasets live elsewhere
# processed_dir = './dpnice/optimized/analysis/{}/'.format(dataset_name)

for dataset in ['acs_income','adult', 'compas', 'heloc']:  # add more datasets if needed
    root = './dpnice/mia_explainer/visualized/{}/NN/'.format(dataset)
    csv_path = '{}aggregated_stats.csv'.format(root)
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_cols = [
        "method_name", "average_distance", "std_distance",
        "average_reidentification_rate", "std_reidentification_rate",
        "success_rate"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    keep = df[
        [
            "method_name",
            "average_distance",
            "std_distance",
            "average_reidentification_rate",
            "std_reidentification_rate",
            "success_rate",
        ]
    ].copy()

    keep["dataset"] = dataset
    keep["distance"] = keep.apply(
        lambda r: f"{r.average_distance:.3f} ± {r.std_distance:.3f}", axis=1
    )
    keep["reid_rate"] = keep.apply(
        lambda r: f"{r.average_reidentification_rate:.3f} ± {r.std_reidentification_rate:.3f}",
        axis=1,
    )

    rows.append(
        keep[
            [
                "dataset",
                "method_name",
                "distance",
                "reid_rate",
                "success_rate",
            ]
        ]
    )

# combine and export LaTeX
if rows:
    table = pd.concat(rows, ignore_index=True)

    # collapse duplicate dataset labels so each dataset name appears once across its methods
    table.loc[table.duplicated(subset=["dataset"]), "dataset"] = ""

    # Escape underscores in string columns for LaTeX
    for col in table.select_dtypes(include=['object']).columns:
        table[col] = table[col].astype(str).str.replace('_', r'\_')

    latex = table.to_latex(
        index=False,
        column_format="lllrr",
        header=[
            "dataset",
            "method",
            "avg\\_distance ± std",
            "avg\\_reid\\_rate ± std",
            "success\\_rate",
        ],
        float_format="%.3f",
        escape=False,
    )

    out_path = Path("./dpnice/mia_explainer/visualized/cf_analysis.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)
    print(f"LaTeX table written to {out_path}")
else:
    print("No aggregated_stats.csv files found for the configured datasets")






 ##################
 # from pathlib import Path
# import pandas as pd
# import numpy as np

# rows = []
# # root = Path("dataset")  # adjust if your datasets live elsewhere
# # processed_dir = './dpnice/optimized/analysis/{}/'.format(dataset_name)

# for dataset in ['acs_income','adult', 'compas', 'heloc']:  # add more datasets if needed
#     root = './dpnice/mia_explainer/visualized/{}/NN/'.format(dataset)
#     csv_path = '{}aggregated_stats.csv'.format(root)
#     df = pd.read_csv(csv_path)
    
#     # Ensure required columns exist
#     required_cols = [
#         "method_name", "average_distance", "std_distance", "average_CF_min_dist",
#         "std_CF_min_dist", "average_reidentification_rate", "std_reidentification_rate",
#         "success_rate"
#     ]
#     for col in required_cols:
#         if col not in df.columns:
#             df[col] = np.nan
    
#     keep = df[
#         [
#             "method_name",
#             "average_distance",
#             "std_distance",
#             "average_CF_min_dist",
#             "std_CF_min_dist",
#             "average_reidentification_rate",
#             "std_reidentification_rate",
#             "success_rate",
#         ]
#     ].copy()

#     keep["dataset"] = dataset
#     keep["distance"] = keep.apply(
#         lambda r: f"{r.average_distance:.3f} ± {r.std_distance:.3f}", axis=1
#     )
#     keep["reid_rate"] = keep.apply(
#         lambda r: f"{r.average_reidentification_rate:.3f} ± {r.std_reidentification_rate:.3f}",
#         axis=1,
#     )

#     rows.append(
#         keep[
#             [
#                 "dataset",
#                 "method_name",
#                 "distance",
#                 "average_CF_min_dist",
#                 "reid_rate",
#                 "success_rate",
#             ]
#         ]
#     )

# # combine and export LaTeX
# if rows:
#     table = pd.concat(rows, ignore_index=True)

#     # collapse duplicate dataset labels so each dataset name appears once across its methods
#     table.loc[table.duplicated(subset=["dataset"]), "dataset"] = ""

#     latex = table.to_latex(
#         index=False,
#         column_format="llrrrr",
#         header=[
#             "dataset",
#             "method",
#             "avg_distance ± std",
#             "avg_CF_min_dist ± std",
#             "avg_reid_rate ± std",
#             "success_rate",
#         ],
#         float_format="%.3f",
#     )

#     out_path = Path("./dpnice/mia_explainer/visualized/cf_analysis.tex")
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_text(latex)
#     print(f"LaTeX table written to {out_path}")
# else:
#     print("No aggregated_stats.csv files found for the configured datasets")   