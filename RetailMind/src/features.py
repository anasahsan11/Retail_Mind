
import pandas as pd


def engineer_features(df):
    """
    Create domain-specific features for customer churn prediction.
    """

    df = df.copy()

    # Convert TotalCharges into numeric format
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # 1. Tenure groups
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, float("inf")],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-60 months",
            "60+ months",
        ],
    ).astype("object")

    # 2. Average charges per month of tenure
    df["avg_charge_per_tenure"] = (
        df["TotalCharges"] / df["tenure"].replace(0, 1)
    )

    # 3. Count subscribed services
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    available_services = [
        col for col in service_columns if col in df.columns
    ]

    df["services_count"] = (
        df[available_services].eq("Yes").sum(axis=1)
    )

    # 4. Contract flags
    df["is_month_to_month"] = (
        df["Contract"] == "Month-to-month"
    ).astype(int)

    df["is_one_year_contract"] = (
        df["Contract"] == "One year"
    ).astype(int)

    df["is_two_year_contract"] = (
        df["Contract"] == "Two year"
    ).astype(int)

    # 5. Service flags
    df["has_online_security"] = (
        df["OnlineSecurity"] == "Yes"
    ).astype(int)

    df["has_tech_support"] = (
        df["TechSupport"] == "Yes"
    ).astype(int)

    # 6. Payment method flag
    df["uses_electronic_check"] = (
        df["PaymentMethod"] == "Electronic check"
    ).astype(int)

    # 7. Monthly charges per service
    df["monthly_charge_per_service"] = (
        df["MonthlyCharges"] / (df["services_count"] + 1)
    )

    # Remove customer ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df
