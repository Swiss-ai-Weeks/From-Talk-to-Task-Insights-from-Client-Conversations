
# Data for the Challenge "From Talk to Task: Insights from Client Conversations"
This respository is part of the Zurich hackathon of [Swiss {ai} Week](https://swiss-ai-weeks.ch/) happening on 26/27 September 2025.

By accessing or using the data provided, you agree to the following terms and conditions.

## Terms and Conditions
> The data is provided solely for the purpose of participating in the hackathon event held in Zurich, Switzerland, and for developing solutions directly related to the specific challenge you have selected. You are strictly prohibited from using the Data for any other purpose, including but not limited to:
> - Commercial use.
> - Research or development outside the scope of this hackathon challenge.
> - Personal use or any other unauthorized activities.
>
> The data is provided "as is" without any warranties, express or implied, including but not limited to, warranties of merchantability, fitness for a particular purpose, or non-infringement. The hackathon organizers do not guarantee the accuracy, completeness, or reliability of the data.
>
> Immediately following the conclusion of the hackathon event, you are obligated to permanently and securely delete all copies of the data, including any derived or processed data, from all your devices, storage media, and systems. 
>
> By using or downloading this data you adhere to the terms and conditions specified in the file DISCLAIMER.md
## Source of Data
The data of this respository has been provided by [UBS](https://www.ubs.com/) submitting the challenge


## Labels

The following permitted task labels are:


  

       [ "plan_contact",

        "schedule_meeting",

        "update_contact_info_non_postal",

        "update_contact_info_postal_address",

        "update_kyc_activity",

        "update_kyc_origin_of_assets",

        "update_kyc_purpose_of_businessrelation",

        "update_kyc_total_assets" ]

## Evaluation
```python
from typing import List
import numpy as np

def evaluate_predictions(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> float:
    """Evaluation for synthetic call transcript classification.

    Args:
        y_true (List[List[str]]): True labels for all samples. Each inner list contains
            true task labels for one sample.
        y_pred (List[List[str]]): Predicted labels for all samples. Each inner list
            contains predicted task labels for one sample.

    Returns:
        float: Score between 0.0 and 1.0 (higher is better).
    """

    ALLOWED_LABELS = [
        "plan_contact",
        "schedule_meeting",
        "update_contact_info_non_postal",
        "update_contact_info_postal_address",
        "update_kyc_activity",
        "update_kyc_origin_of_assets",
        "update_kyc_purpose_of_businessrelation",
        "update_kyc_total_assets",
    ]
    LABEL_TO_IDX = {label: idx for idx, label in enumerate(ALLOWED_LABELS)}

    FN_PENALTY = 2.0
    FP_PENALTY = 1.0

    def _process_sample_labels(sample_labels: List[str], sample_name: str) -> List[str]:
        """Process and validate labels for one sample."""
        if not isinstance(sample_labels, list):
            raise ValueError(
                f"{sample_name} must be a list of strings, got {type(sample_labels)}"
            )

        # Check for duplicates
        unique_labels = []
        seen = set()
        for label in sample_labels:
            if not isinstance(label, str):
                raise ValueError(
                    f"{sample_name} contains non-string label: {label} "
                    f"(type: {type(label)})"
                )
            if label in seen:
                raise ValueError(f"{sample_name} contains duplicate label: '{label}'")
            else:
                seen.add(label)
                unique_labels.append(label)

        # Check for invalid labels
        valid_labels = []
        for label in unique_labels:
            if label not in ALLOWED_LABELS:
                raise ValueError(
                    f"{sample_name} contains invalid label: '{label}'. "
                    f"Allowed labels: {ALLOWED_LABELS}"
                )
            else:
                valid_labels.append(label)

        return valid_labels

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have same length. Got "
            f"{len(y_true)} vs {len(y_pred)}"
        )

    n_samples = len(y_true)
    n_labels = len(ALLOWED_LABELS)

    # Convert string labels to binary arrays
    y_true_binary = np.zeros((n_samples, n_labels), dtype=int)
    y_pred_binary = np.zeros((n_samples, n_labels), dtype=int)

    # Process true labels
    for i, sample_labels in enumerate(y_true):
        processed_labels = _process_sample_labels(sample_labels, f"y_true[{i}]")
        for label in processed_labels:
            label_idx = LABEL_TO_IDX[label]
            y_true_binary[i, label_idx] = 1

    # Process predicted labels
    for i, sample_labels in enumerate(y_pred):
        processed_labels = _process_sample_labels(sample_labels, f"y_pred[{i}]")
        for label in processed_labels:
            label_idx = LABEL_TO_IDX[label]
            y_pred_binary[i, label_idx] = 1

    # Count errors per sample
    false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0), axis=1)
    false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1), axis=1)

    # Weighted error using custom penalties
    weighted_errors = FN_PENALTY * false_negatives + FP_PENALTY * false_positives

    # Calculate max possible error per sample
    max_errors_per_sample = FN_PENALTY * np.sum(y_true_binary, axis=1) + FP_PENALTY * (
        n_labels - np.sum(y_true_binary, axis=1)
    )

    # Per-sample normalization then average
    per_sample_scores = np.where(
        max_errors_per_sample > 0,
        1.0 - (weighted_errors / max_errors_per_sample),
        1.0,  # Perfect score when no penalties apply
    )
    final_score = float(np.mean(per_sample_scores))

    # Clamp final result to [0.0, 1.0] for safety against numeric issues
    return max(0.0, min(1.0, final_score))

y_true = [
    ["schedule_meeting", "update_contact_info_non_postal", "update_contact_info_postal_address"],
    [],
    ["update_kyc_activity"],
]

y_pred = [
    ["schedule_meeting", "update_contact_info_non_postal"],
    [],
    ["plan_contact", "update_kyc_total_assets"],
]

score = evaluate_predictions(y_true, y_pred)
print(f"Score: {score:.3f}")  # Score: 0.791
