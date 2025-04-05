import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series import FeatureTransformer, DefaultFeatures, TabPFNTimeSeriesPredictor, TabPFNMode
from tabpfn_time_series.plot import plot_actual_ts, plot_pred_and_actual_ts
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# --- Configuration ---
DATA_FILE = 'synthetic_electronics_single_country_sales_LONG_2024_2025.csv'
PREDICTION_LENGTH = 30

# Promotion encoding mapping
PROMOTION_MAPPING = {
    'No Promotion': 0,
    'SpringSale': 1,
    'SummerDeals': 1,
    'Back2School': 1,
    'BlackFridayWeek': 1,
    'HolidaySale': 1,
    'YearEndClearance': 1
}

# Features to use in time series model
SELECTED_FEATURES = [
    DefaultFeatures.add_running_index,
    DefaultFeatures.add_calendar_features
]


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the sales data.
    
    Args:
        file_path (str): Path to the CSV data file
        
    Returns:
        DataFrame: Preprocessed pandas DataFrame
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Clean and transform data
    df['ActivePromotion'] = df['ActivePromotion'].fillna('No Promotion')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Aggregate sales by category, timestamp, and promotion
    df = df.groupby(['Category', 'Timestamp', 'ActivePromotion'])['Sales(USD)'].sum()
    df = df.reset_index()
    
    # Apply promotion mapping and rename columns
    df['ActivePromotion'] = df['ActivePromotion'].map(PROMOTION_MAPPING)
    df = df.rename(columns={'Sales(USD)': 'target', 'ActivePromotion': 'Promotion'})
    
    return df


def convert_to_time_series_format(df):
    """
    Convert pandas DataFrame to TimeSeriesDataFrame format.
    
    Args:
        df (DataFrame): Preprocessed pandas DataFrame
        
    Returns:
        TimeSeriesDataFrame: AutoGluon time series format
    """
    tsdf = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column='Category',
        timestamp_column='Timestamp',
    )
    return tsdf


def prepare_train_test_data(tsdf, prediction_length):
    """
    Split data into training and test sets.
    
    Args:
        tsdf (TimeSeriesDataFrame): Time series data
        prediction_length (int): Number of time steps to predict
        
    Returns:
        tuple: (train_tsdf, test_tsdf, test_tsdf_ground_truth)
    """
    # Split data into train and test (ground truth)
    train_tsdf, test_tsdf_ground_truth = tsdf.train_test_split(prediction_length=prediction_length)
    
    # Generate test data structure without values for prediction
    test_tsdf = generate_test_X(train_tsdf, prediction_length)
    
    return train_tsdf, test_tsdf, test_tsdf_ground_truth


def add_time_series_features(train_tsdf, test_tsdf, selected_features):
    """
    Add time series features to the data.
    
    Args:
        train_tsdf (TimeSeriesDataFrame): Training data
        test_tsdf (TimeSeriesDataFrame): Test data
        selected_features (list): Features to add
        
    Returns:
        tuple: (train_tsdf, test_tsdf) with added features
    """
    return FeatureTransformer.add_features(
        train_tsdf, test_tsdf, selected_features
    )


def make_predictions(train_tsdf, test_tsdf):
    """
    Make time series predictions using TabPFN model.
    
    Args:
        train_tsdf (TimeSeriesDataFrame): Training data with features
        test_tsdf (TimeSeriesDataFrame): Test data with features
        
    Returns:
        TimeSeriesDataFrame: Predictions
    """
    predictor = TabPFNTimeSeriesPredictor(
        tabpfn_mode=TabPFNMode.CLIENT,
    )
    
    return predictor.predict(train_tsdf, test_tsdf)


def visualize_time_series(train_tsdf, test_tsdf_ground_truth):
    """
    Visualize actual time series data.
    
    Args:
        train_tsdf (TimeSeriesDataFrame): Training data
        test_tsdf_ground_truth (TimeSeriesDataFrame): Test data with ground truth
    """
    plot_actual_ts(train_tsdf, test_tsdf_ground_truth)


def visualize_predictions(train_tsdf, test_tsdf_ground_truth, pred):
    """
    Visualize predictions against actual data.
    
    Args:
        train_tsdf (TimeSeriesDataFrame): Training data
        test_tsdf_ground_truth (TimeSeriesDataFrame): Test data with ground truth
        pred (TimeSeriesDataFrame): Predictions
    """
    # Drop promotion column if it exists
    train_clean = train_tsdf.drop(columns=['Promotion'], errors='ignore')
    test_clean = test_tsdf_ground_truth.drop(columns=['Promotion'], errors='ignore')
    
    plot_pred_and_actual_ts(
        train=train_clean,
        test=test_clean,
        pred=pred
    )


def calculate_forecast_metrics(pred, test_tsdf_ground_truth):
    """
    Calculate and display forecast accuracy metrics.
    
    Args:
        pred (TimeSeriesDataFrame): Predicted values
        test_tsdf_ground_truth (TimeSeriesDataFrame): Actual values
    
    Returns:
        DataFrame: Metrics by category
    """
    # Convert data to DataFrames
    pred_df = pred.to_data_frame()
    test_df = test_tsdf_ground_truth.to_data_frame()
    
    # Get unique item IDs
    item_ids = test_df.index.get_level_values('item_id').unique()
    
    # Calculate metrics for each item
    all_metrics = {}
    for item_id in item_ids:
        # Filter for this specific item
        pred_item = pred_df[pred_df.index.get_level_values('item_id') == item_id]
        test_item = test_df[test_df.index.get_level_values('item_id') == item_id]
        
        # Align indices
        common_indices = pred_item.index.intersection(test_item.index)
        
        # Calculate metrics
        y_true = test_item.loc[common_indices]['target']
        y_pred = pred_item.loc[common_indices]['target']
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        all_metrics[item_id] = {'RMSE': rmse, 'MAPE': mape}
    
    # Create summary dataframe
    metrics_df = pd.DataFrame(all_metrics).T
    
    # Print results
    print("\nSummary of Metrics:")
    print(metrics_df)
    print(f"Average RMSE: {metrics_df['RMSE'].mean():.2f}")
    print(f"Average MAPE: {metrics_df['MAPE'].mean():.2f}%")
    
    return metrics_df


def main():
    """
    Main function to execute the complete time series analysis pipeline.
    """
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    df = load_and_preprocess_data(DATA_FILE)
    
    # Step 2: Convert to time series format
    print("Step 2: Converting to time series format...")
    tsdf = convert_to_time_series_format(df)
    
    # Step 3: Prepare train/test splits
    print("Step 3: Preparing training and test data...")
    train_tsdf, test_tsdf, test_tsdf_ground_truth = prepare_train_test_data(
        tsdf, PREDICTION_LENGTH
    )
    
    # Step 4: Visualize actual time series data
    print("Step 4: Visualizing time series data...")
    visualize_time_series(train_tsdf, test_tsdf_ground_truth)
    
    # Step 5: Add time series features
    print("Step 5: Adding time series features...")
    train_tsdf_with_features, test_tsdf_with_features = add_time_series_features(
        train_tsdf, test_tsdf, SELECTED_FEATURES
    )
    
    # Step 6: Make predictions
    print("Step 6: Making predictions...")
    predictions = make_predictions(train_tsdf_with_features, test_tsdf_with_features)
    
    # Step 7: Visualize predictions
    print("Step 7: Visualizing predictions...")
    visualize_predictions(train_tsdf, test_tsdf_ground_truth, predictions)
    
    # Step 8: Calculate and display metrics
    print("Step 8: Calculating forecast metrics...")
    metrics = calculate_forecast_metrics(predictions, test_tsdf_ground_truth)
    
    return train_tsdf, test_tsdf_ground_truth, predictions, metrics


if __name__ == "__main__":
    train_tsdf, test_tsdf_ground_truth, predictions, metrics = main()