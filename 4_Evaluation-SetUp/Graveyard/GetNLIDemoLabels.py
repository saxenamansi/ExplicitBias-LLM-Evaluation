"""Join narrative projections with NLI demographic labels into wide format."""

import pandas as pd

PROJECTIONS_PATH = "ProjectionValidation/NarrativeProjections.csv"
NLI_PATH         = "../Data/SentenceNLI-SV-Full.csv"
OUT_PATH         = "ProjectionValidation/NarrativeProjections-withDemographics.csv"
DEMO_CATS        = ["AGE", "GENDER", "PERSPECTIVE", "RELATIONSHIP", "MYTH"]

def build_wide_labels(nli):
    """Pivot NLI labels to one row per narrative, one column per myth_type."""
    rows = []
    for cat in DEMO_CATS:
        cat_df = nli[nli["myth_category"] == cat][
            ["narrative_index", "myth_type", "overall_label"]
        ]
        rows.append(
            cat_df.pivot(index="narrative_index", columns="myth_type", values="overall_label")
        )
    return pd.concat(rows, axis=1).reset_index()

def main():
    proj_df = pd.read_csv(PROJECTIONS_PATH)
    nli     = pd.read_csv(NLI_PATH)
    wide    = build_wide_labels(nli)
    proj_df.merge(wide, on="narrative_index").to_csv(OUT_PATH, index=False)

if __name__ == "__main__":
    main()