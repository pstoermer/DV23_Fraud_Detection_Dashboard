# Fraud Detection Dashboard

## Overview

The Fraud Detection Dashboard provides interactive tools for analyzing and detecting fraudulent insurance claims. It
leverages the Kaggle [Car_Insurance_Fraud](https://www.kaggle.com/datasets/incarnyx/car-insurance-fraud) dataset of car
insurance claims to visualize trends, explore attribute relationships, and identify potential fraud patterns.

## Features

- **Overall Trends and Attribute Distribution:** This section provides an overview of the dataset, including key numbers
  and a timeline of fraudulent and non-fraudulent cases over time. Users can toggle between fraud and non-fraud cases
  and adjust attribute combinations to explore different trends and distributions.

- **Relationship between Attribute Pairs and Fraudulent Cases:** Users can investigate the association between attribute
  pairs and fraud in this section. The heatmap visualizes the intensity of the relationship, while the stacked bar chart
  shows the proportion of fraud cases within each attribute pair. Combination controls allow users to dynamically change
  the attribute pairs for analysis.

- **Multivariate Attribute Patterns for Fraud Identification:** This section enables users to identify multivariate
  attribute patterns that indicate potential fraud. Users can explore parallel coordinates plots to visualize the
  interactions between multiple attributes and their impact on fraud detection.

## Installation

1. Clone the repository: `git clone https://github.com/your-username/fraud-detection-dashboard.git`
2. Navigate to the project directory: `cd fraud_detection_dashboard`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Start the application: `python app.py`
5. Open your web browser and go to `http://127.0.0.1:8050/` to access the Fraud Detection Dashboard.

## Dataset

The dashboard utilizes the Car Insurance Fraud Dataset obtained from Kaggle. It contains 11.6k rows with various
attributes such as month, day of the week, make of the vehicle, age of the policyholder, and the claim size. The dataset
is included in the repository as `Dataset.xlsx`.


## License

This project is licensed under the [MIT License](LICENSE).


