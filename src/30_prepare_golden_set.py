import pandas as pd
import os


# ============================================================
# STEP 1: File paths
# ============================================================

INPUT_FILE = "evaluation/golden_set.csv"
OUTPUT_FILE = "evaluation/golden_set_labeled.csv"


# ============================================================
# STEP 2: Load golden set
# ============================================================

print("Loading golden set...")

df = pd.read_csv(INPUT_FILE)

print("Rows found:", len(df))

if len(df) != 200:
    raise ValueError(
        f"Expected 200 rows, but found {len(df)}"
    )


# ============================================================
# STEP 3: Intent labels for the 200 examples
# ============================================================

labels = [

    # --------------------------------------------------------
    # Examples 1 - 20
    # --------------------------------------------------------

    "general_support",
    "battery_issue",
    "app_or_itunes_issue",
    "device_functionality_issue",
    "software_update_issue",
    "software_update_issue",
    "device_functionality_issue",
    "general_support",
    "battery_issue",
    "device_functionality_issue",
    "device_functionality_issue",
    "general_support",
    "information_or_how_to",
    "device_functionality_issue",
    "connectivity_issue",
    "device_functionality_issue",
    "software_update_issue",
    "other",
    "connectivity_issue",
    "general_support",


    # --------------------------------------------------------
    # Examples 21 - 40
    # --------------------------------------------------------

    "app_or_itunes_issue",
    "software_update_issue",
    "app_or_itunes_issue",
    "device_functionality_issue",
    "performance_freezing",
    "device_functionality_issue",
    "app_or_itunes_issue",
    "other",
    "software_update_issue",
    "general_support",
    "performance_freezing",
    "general_support",
    "software_update_issue",
    "software_update_issue",
    "software_update_issue",
    "device_functionality_issue",
    "software_update_issue",
    "battery_issue",
    "software_update_issue",
    "general_support",


    # --------------------------------------------------------
    # Examples 41 - 60
    # --------------------------------------------------------

    "battery_issue",
    "performance_freezing",
    "connectivity_issue",
    "app_or_itunes_issue",
    "information_or_how_to",
    "device_functionality_issue",
    "account_or_billing_issue",
    "connectivity_issue",
    "performance_freezing",
    "information_or_how_to",
    "account_or_billing_issue",
    "app_or_itunes_issue",
    "device_functionality_issue",
    "battery_issue",
    "information_or_how_to",
    "performance_freezing",
    "account_or_billing_issue",
    "connectivity_issue",
    "general_support",
    "other",


    # --------------------------------------------------------
    # Examples 61 - 80
    # --------------------------------------------------------

    "device_functionality_issue",
    "device_functionality_issue",
    "device_functionality_issue",
    "device_functionality_issue",
    "device_functionality_issue",
    "general_support",
    "app_or_itunes_issue",
    "software_update_issue",
    "device_functionality_issue",
    "device_functionality_issue",
    "performance_freezing",
    "app_or_itunes_issue",
    "battery_issue",
    "general_support",
    "software_update_issue",
    "device_functionality_issue",
    "device_functionality_issue",
    "other",
    "software_update_issue",
    "software_update_issue",


    # --------------------------------------------------------
    # Examples 81 - 100
    # --------------------------------------------------------

    "performance_freezing",
    "device_functionality_issue",
    "information_or_how_to",
    "software_update_issue",
    "connectivity_issue",
    "general_support",
    "device_functionality_issue",
    "software_update_issue",
    "general_support",
    "device_functionality_issue",
    "software_update_issue",
    "software_update_issue",
    "account_or_billing_issue",
    "software_update_issue",
    "battery_issue",
    "account_or_billing_issue",
    "device_functionality_issue",
    "device_functionality_issue",
    "information_or_how_to",
    "general_support",


    # --------------------------------------------------------
    # Examples 101 - 120
    # --------------------------------------------------------

    "device_functionality_issue",
    "software_update_issue",
    "information_or_how_to",
    "performance_freezing",
    "device_functionality_issue",
    "software_update_issue",
    "device_functionality_issue",
    "device_functionality_issue",
    "general_support",
    "device_functionality_issue",
    "performance_freezing",
    "performance_freezing",
    "software_update_issue",
    "performance_freezing",
    "battery_issue",
    "software_update_issue",
    "information_or_how_to",
    "software_update_issue",
    "device_functionality_issue",
    "battery_issue",


    # --------------------------------------------------------
    # Examples 121 - 140
    # --------------------------------------------------------

    "battery_issue",
    "software_update_issue",
    "account_or_billing_issue",
    "software_update_issue",
    "device_functionality_issue",
    "battery_issue",
    "general_support",
    "software_update_issue",
    "device_functionality_issue",
    "software_update_issue",
    "software_update_issue",
    "software_update_issue",
    "information_or_how_to",
    "device_functionality_issue",
    "battery_issue",
    "battery_issue",
    "software_update_issue",
    "device_functionality_issue",
    "performance_freezing",
    "software_update_issue",


    # --------------------------------------------------------
    # Examples 141 - 160
    # --------------------------------------------------------

    "device_functionality_issue",
    "other",
    "software_update_issue",
    "device_functionality_issue",
    "battery_issue",
    "device_functionality_issue",
    "software_update_issue",
    "device_functionality_issue",
    "software_update_issue",
    "battery_issue",
    "software_update_issue",
    "software_update_issue",
    "app_or_itunes_issue",
    "device_functionality_issue",
    "battery_issue",
    "app_or_itunes_issue",
    "software_update_issue",
    "device_functionality_issue",
    "battery_issue",
    "device_functionality_issue",


    # --------------------------------------------------------
    # Examples 161 - 180
    # --------------------------------------------------------

    "software_update_issue",
    "device_functionality_issue",
    "app_or_itunes_issue",
    "performance_freezing",
    "app_or_itunes_issue",
    "software_update_issue",
    "software_update_issue",
    "device_functionality_issue",
    "software_update_issue",
    "software_update_issue",
    "performance_freezing",
    "general_support",
    "software_update_issue",
    "performance_freezing",
    "performance_freezing",
    "app_or_itunes_issue",
    "performance_freezing",
    "software_update_issue",
    "software_update_issue",
    "device_functionality_issue",


    # --------------------------------------------------------
    # Examples 181 - 200
    # --------------------------------------------------------

    "software_update_issue",
    "device_functionality_issue",
    "device_functionality_issue",
    "performance_freezing",
    "general_support",
    "other",
    "account_or_billing_issue",
    "software_update_issue",
    "device_functionality_issue",
    "performance_freezing",
    "general_support",
    "general_support",
    "performance_freezing",
    "software_update_issue",
    "device_functionality_issue",
    "other",
    "information_or_how_to",
    "device_functionality_issue",
    "battery_issue",
    "battery_issue",
]


# ============================================================
# STEP 4: Verify number of labels
# ============================================================

print("Number of labels:", len(labels))

if len(labels) != 200:
    raise ValueError(
        f"Expected 200 labels, but got {len(labels)}"
    )


# ============================================================
# STEP 5: Valid intents
# ============================================================

valid_intents = {
    "software_update_issue",
    "battery_issue",
    "performance_freezing",
    "app_or_itunes_issue",
    "connectivity_issue",
    "device_functionality_issue",
    "account_or_billing_issue",
    "information_or_how_to",
    "general_support",
    "other",
}


# ============================================================
# STEP 6: Check labels
# ============================================================

invalid_labels = set(labels) - valid_intents

if invalid_labels:
    raise ValueError(
        f"Invalid labels found: {invalid_labels}"
    )


# ============================================================
# STEP 7: Add labels to dataframe
# ============================================================

df["intent"] = labels

df["labeling_notes"] = (
    "Reviewed using the golden-set labeling guide."
)


# ============================================================
# STEP 8: Save labeled golden set
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# STEP 9: Display results
# ============================================================

print()
print("=" * 60)
print("GOLDEN SET PREPARATION COMPLETE")
print("=" * 60)

print("Total examples:", len(df))

print("Output file:")
print(OUTPUT_FILE)

print()
print("Intent distribution:")
print("-" * 60)

print(
    df["intent"]
    .value_counts()
    .sort_index()
)


# ============================================================
# STEP 10: Check missing labels
# ============================================================

missing_labels = df["intent"].isna().sum()

print()
print("Missing labels:", missing_labels)

if missing_labels == 0:
    print("SUCCESS: All 200 examples have labels.")
else:
    print("WARNING: Some examples have missing labels.")


print()
print("Done.")