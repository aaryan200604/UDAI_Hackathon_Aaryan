"""
Time-Series Forecasting Module
Predicts future enrolment volumes and flags expected anomalies.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta


class SimpleForecaster:
    """
    Simple time-series forecaster using moving averages and trend decomposition.
    Works without heavy dependencies like Prophet.
    """
    
    def __init__(self, window: int = 7, trend_window: int = 14):
        """
        Initialize the forecaster.
        
        Args:
            window: Window size for moving average
            trend_window: Window size for trend calculation
        """
        self.window = window
        self.trend_window = trend_window
        self.history = None
        self.trend_coef = None
        self.seasonality = None
        
    def fit(self, df: pd.DataFrame, date_col: str = 'date', value_col: str = 'total_enrolments') -> 'SimpleForecaster':
        """
        Fit the forecaster on historical data.
        
        Args:
            df: DataFrame with time series data
            date_col: Name of date column
            value_col: Name of value column
            
        Returns:
            Self for method chaining
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Aggregate by date
        daily = df.groupby(date_col)[value_col].sum().reset_index()
        daily = daily.sort_values(date_col)
        
        self.history = daily.copy()
        self.value_col = value_col
        self.date_col = date_col
        
        # Calculate trend using linear regression
        x = np.arange(len(daily))
        y = daily[value_col].values
        
        # Simple linear regression
        self.trend_coef = np.polyfit(x, y, 1)
        
        # Calculate weekly seasonality (day of week effects)
        daily['dow'] = pd.to_datetime(daily[date_col]).dt.dayofweek
        self.seasonality = daily.groupby('dow')[value_col].mean()
        overall_mean = daily[value_col].mean()
        self.seasonality = self.seasonality / overall_mean
        
        return self
    
    def predict(self, periods: int = 7, return_bounds: bool = True) -> pd.DataFrame:
        """
        Predict future values.
        
        Args:
            periods: Number of periods to forecast
            return_bounds: Whether to return confidence bounds
            
        Returns:
            DataFrame with predictions
        """
        if self.history is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        last_date = self.history[self.date_col].max()
        last_idx = len(self.history) - 1
        
        # Calculate base statistics for uncertainty
        recent = self.history.tail(self.trend_window)[self.value_col]
        std = recent.std()
        
        predictions = []
        for i in range(1, periods + 1):
            future_date = last_date + timedelta(days=i)
            future_idx = last_idx + i
            
            # Trend component
            trend = np.polyval(self.trend_coef, future_idx)
            
            # Seasonality component
            dow = future_date.weekday()
            seasonal_factor = self.seasonality.get(dow, 1.0)
            
            # Final prediction
            predicted = trend * seasonal_factor
            
            # Confidence bounds (wider as we go further)
            uncertainty = std * np.sqrt(i) * 1.96
            
            predictions.append({
                'date': future_date,
                'predicted': max(0, predicted),
                'lower_bound': max(0, predicted - uncertainty),
                'upper_bound': predicted + uncertainty,
                'trend': trend,
                'seasonal_factor': seasonal_factor
            })
        
        return pd.DataFrame(predictions)
    
    def detect_future_anomalies(self, forecast_df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
        """
        Flag periods where forecast suggests potential anomalies.
        
        Args:
            forecast_df: DataFrame from predict()
            threshold: Threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags
        """
        forecast_df = forecast_df.copy()
        
        # Calculate volatility indicator
        avg_value = self.history[self.value_col].mean()
        
        # High volatility if bounds are wide relative to average
        forecast_df['volatility_ratio'] = (forecast_df['upper_bound'] - forecast_df['lower_bound']) / avg_value
        
        # Flag high uncertainty periods
        forecast_df['high_uncertainty'] = forecast_df['volatility_ratio'] > threshold
        
        # Check for significant trend changes
        if len(forecast_df) > 1:
            forecast_df['trend_change'] = forecast_df['predicted'].pct_change().abs()
            forecast_df['anomaly_risk'] = (forecast_df['trend_change'] > 0.2) | forecast_df['high_uncertainty']
        else:
            forecast_df['anomaly_risk'] = forecast_df['high_uncertainty']
        
        return forecast_df


def forecast_by_state(
    df: pd.DataFrame,
    periods: int = 7,
    date_col: str = 'date',
    value_col: str = 'total_enrolments',
    state_col: str = 'state'
) -> pd.DataFrame:
    """
    Generate forecasts for each state.
    
    Args:
        df: DataFrame with time series data
        periods: Number of periods to forecast
        date_col: Name of date column
        value_col: Name of value column
        state_col: Name of state column
        
    Returns:
        DataFrame with forecasts for all states
    """
    all_forecasts = []
    
    for state in df[state_col].unique():
        state_df = df[df[state_col] == state]
        
        if len(state_df) < 7:  # Need minimum data
            continue
        
        forecaster = SimpleForecaster()
        forecaster.fit(state_df, date_col, value_col)
        
        forecast = forecaster.predict(periods)
        forecast[state_col] = state
        forecast = forecaster.detect_future_anomalies(forecast)
        
        all_forecasts.append(forecast)
    
    if not all_forecasts:
        return pd.DataFrame()
    
    return pd.concat(all_forecasts, ignore_index=True)


def forecast_total_enrolments(
    df: pd.DataFrame,
    periods: int = 14,
    date_col: str = 'date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast total enrolments across all regions.
    
    Args:
        df: DataFrame with time series data
        periods: Number of periods to forecast
        date_col: Name of date column
        
    Returns:
        Tuple of (historical data, forecast data)
    """
    # Aggregate by date
    df[date_col] = pd.to_datetime(df[date_col])
    
    if 'total_enrolments' not in df.columns:
        df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    
    daily = df.groupby(date_col)['total_enrolments'].sum().reset_index()
    
    # Fit and predict
    forecaster = SimpleForecaster()
    forecaster.fit(daily, date_col, 'total_enrolments')
    
    forecast = forecaster.predict(periods)
    forecast = forecaster.detect_future_anomalies(forecast)
    
    return daily, forecast


class ExponentialSmoothingForecaster:
    """
    Exponential smoothing forecaster for more robust predictions.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        """
        Initialize with smoothing parameters.
        
        Args:
            alpha: Level smoothing parameter (0-1)
            beta: Trend smoothing parameter (0-1)
        """
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None
        
    def fit(self, values: np.ndarray) -> 'ExponentialSmoothingForecaster':
        """
        Fit the model using Holt's linear method.
        
        Args:
            values: Array of values to fit
            
        Returns:
            Self for method chaining
        """
        n = len(values)
        
        # Initialize
        self.level = values[0]
        self.trend = values[1] - values[0] if n > 1 else 0
        
        # Update
        for t in range(1, n):
            prev_level = self.level
            self.level = self.alpha * values[t] + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        
        # Store for error estimation
        self.values = values
        
        return self
    
    def predict(self, periods: int = 7) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            periods: Number of periods ahead
            
        Returns:
            Array of predictions
        """
        if self.level is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        predictions = []
        for h in range(1, periods + 1):
            predictions.append(self.level + h * self.trend)
        
        return np.array(predictions)


if __name__ == "__main__":
    # Test forecasting
    print("Testing forecasting module...")
    
    # Create sample data
    dates = pd.date_range('2025-01-01', periods=30)
    values = 1000 + np.random.randn(30) * 100 + np.arange(30) * 10  # Upward trend
    
    sample_data = pd.DataFrame({
        'date': dates,
        'total_enrolments': values
    })
    
    # Fit and predict
    forecaster = SimpleForecaster()
    forecaster.fit(sample_data)
    
    forecast = forecaster.predict(periods=7)
    forecast = forecaster.detect_future_anomalies(forecast)
    
    print("\nForecast Results:")
    print(forecast[['date', 'predicted', 'lower_bound', 'upper_bound', 'anomaly_risk']])
