import requests


URL = (
    "http://127.0.0.1:8000/predict"
)


sample_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.50,
    "TotalCharges": 906.00,
}


def main():

    print(
        f"Sending request to {URL}"
    )

    response = requests.post(
        URL,
        json=sample_data,
        timeout=30,
    )

    print(
        f"\nStatus code: "
        f"{response.status_code}"
    )

    print(
        "\nResponse:"
    )

    try:
        print(
            response.json()
        )
    except ValueError:
        print(
            response.text
        )

    response.raise_for_status()


if __name__ == "__main__":
    main()