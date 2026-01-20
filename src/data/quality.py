"""
Enhanced Data Quality Checks Module
Comprehensive data validation and quality assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class DataQualityReport:
    """Container for data quality assessment results."""
    
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []
        self.metadata = {}
        
    def add_check(self, name: str, passed: bool, message: str, details: Dict = None):
        """Add a quality check result."""
        self.checks.append({
            'name': name,
            'passed': passed,
            'message': message,
            'details': details or {}
        })
        
        if not passed:
            self.errors.append(message)
    
    def add_warning(self, message: str):
        """Add a warning."""
        self.warnings.append(message)
    
    def is_valid(self) -> bool:
        """Check if all critical checks passed."""
        return len(self.errors) == 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'valid': self.is_valid(),
            'total_checks': len(self.checks),
            'passed': sum(1 for c in self.checks if c['passed']),
            'failed': sum(1 for c in self.checks if not c['passed']),
            'warnings': len(self.warnings),
            'checks': self.checks,
            'errors': self.errors,
            'warnings_list': self.warnings,
            'metadata': self.metadata
        }
    
    def __repr__(self):
        status = "PASSED" if self.is_valid() else "FAILED"
        return f"DataQualityReport({status}: {len(self.checks)} checks, {len(self.warnings)} warnings)"


class DataQualityValidator:
    """Comprehensive data quality validator."""
    
    def __init__(self):
        self.report = DataQualityReport()
    
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Validate that expected columns are present."""
        missing_cols = set(expected_columns) - set(df.columns)
        
        if missing_cols:
            self.report.add_check(
                'schema_validation',
                False,
                f"Missing required columns: {missing_cols}",
                {'missing_columns': list(missing_cols)}
            )
            return False
        else:
            self.report.add_check(
                'schema_validation',
                True,
                "All required columns present"
            )
            return True
    
    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.1) -> bool:
        """Check for excessive missing values."""
        missing_pct = df.isnull().sum() / len(df)
        problematic = missing_pct[missing_pct > threshold]
        
        if len(problematic) > 0:
            self.report.add_warning(
                f"Columns with >{threshold*100}% missing values: {problematic.to_dict()}"
            )
            self.report.add_check(
                'missing_values',
                False,
                f"{len(problematic)} columns have excessive missing values",
                {'problematic_columns': problematic.to_dict()}
            )
            return False
        else:
            self.report.add_check(
                'missing_values',
                True,
                "Missing values within acceptable range"
            )
            return True
    
    def check_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> bool:
        """Check for duplicate records."""
        if subset:
            duplicates = df.duplicated(subset=subset).sum()
        else:
            duplicates = df.duplicated().sum()
        
        if duplicates > 0:
            self.report.add_warning(
                f"Found {duplicates} duplicate records"
            )
            self.report.add_check(
                'duplicates',
                False,
                f"{duplicates} duplicate records found",
                {'count': int(duplicates)}
            )
            return False
        else:
            self.report.add_check(
                'duplicates',
                True,
                "No duplicate records found"
            )
            return True
    
    def check_date_continuity(self, df: pd.DataFrame, date_col: str = 'date') -> bool:
        """Check for gaps in date sequence."""
        if date_col not in df.columns:
            return True
        
        df_sorted = df.sort_values(date_col)
        dates = pd.to_datetime(df_sorted[date_col])
        
        # Get unique dates
        unique_dates = dates.unique()
        date_range = pd.date_range(start=unique_dates.min(), end=unique_dates.max(), freq='D')
        
        missing_dates = set(date_range) - set(unique_dates)
        
        if len(missing_dates) > 0:
            self.report.add_warning(
                f"Found {len(missing_dates)} missing dates in sequence"
            )
            self.report.add_check(
                'date_continuity',
                False,
                f"{len(missing_dates)} dates missing from sequence",
                {'missing_dates_count': len(missing_dates)}
            )
            return False
        else:
            self.report.add_check(
                'date_continuity',
                True,
                "Date sequence is continuous"
            )
            return True
    
    def check_value_ranges(self, df: pd.DataFrame, columns: Dict[str, Tuple[float, float]]) -> bool:
        """Check if values are within expected ranges."""
        all_valid = True
        
        for col, (min_val, max_val) in columns.items():
            if col not in df.columns:
                continue
            
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            
            if out_of_range > 0:
                all_valid = False
                self.report.add_warning(
                    f"{col}: {out_of_range} values out of range [{min_val}, {max_val}]"
                )
        
        self.report.add_check(
            'value_ranges',
            all_valid,
            "All values within expected ranges" if all_valid else "Some values out of range"
        )
        
        return all_valid
    
    def check_negative_values(self, df: pd.DataFrame, columns: List[str]) -> bool:
        """Check for unexpected negative values."""
        all_valid = True
        
        for col in columns:
            if col not in df.columns:
                continue
            
            negative_count = (df[col] < 0).sum()
            
            if negative_count > 0:
                all_valid = False
                self.report.add_warning(
                    f"{col}: {negative_count} negative values found"
                )
        
        self.report.add_check(
            'negative_values',
            all_valid,
            "No unexpected negative values" if all_valid else "Some negative values found"
        )
        
        return all_valid
    
    def check_data_distribution(self, df: pd.DataFrame, column: str) -> bool:
        """Check for unusual data distribution (outliers)."""
        if column not in df.columns:
            return True
        
        values = df[column].dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        outliers = ((values < lower_bound) | (values > upper_bound)).sum()
        outlier_pct = outliers / len(values) * 100
        
        if outlier_pct > 5:
            self.report.add_warning(
                f"{column}: {outlier_pct:.1f}% outliers detected"
            )
        
        self.report.add_check(
            f'distribution_{column}',
            outlier_pct <= 10,
            f"Distribution check: {outlier_pct:.1f}% outliers",
            {'outlier_percentage': outlier_pct}
        )
        
        return outlier_pct <= 10
    
    def check_categorical_consistency(self, df: pd.DataFrame, column: str, expected_values: Optional[List] = None) -> bool:
        """Check categorical column consistency."""
        if column not in df.columns:
            return True
        
        unique_values = df[column].unique()
        
        if expected_values:
            unexpected = set(unique_values) - set(expected_values)
            if unexpected:
                self.report.add_warning(
                    f"{column}: Unexpected values found: {unexpected}"
                )
                self.report.add_check(
                    f'categorical_{column}',
                    False,
                    f"Unexpected categorical values in {column}",
                    {'unexpected_values': list(unexpected)}
                )
                return False
        
        self.report.add_check(
            f'categorical_{column}',
            True,
            f"{column}: {len(unique_values)} unique values"
        )
        return True
    
    def check_data_freshness(self, df: pd.DataFrame, date_col: str = 'date', max_age_days: int = 30) -> bool:
        """Check if data is recent enough."""
        if date_col not in df.columns:
            return True
        
        latest_date = pd.to_datetime(df[date_col]).max()
        age_days = (datetime.now() - latest_date).days
        
        is_fresh = age_days <= max_age_days
        
        self.report.add_check(
            'data_freshness',
            is_fresh,
            f"Latest data is {age_days} days old",
            {'age_days': age_days, 'latest_date': str(latest_date)}
        )
        
        if not is_fresh:
            self.report.add_warning(f"Data may be stale ({age_days} days old)")
        
        return is_fresh
    
    def check_completeness(self, df: pd.DataFrame) -> bool:
        """Check overall data completeness."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_pct = (1 - missing_cells / total_cells) * 100
        
        is_complete = completeness_pct >= 95
        
        self.report.add_check(
            'completeness',
            is_complete,
            f"Data completeness: {completeness_pct:.2f}%",
            {'completeness_percentage': completeness_pct}
        )
        
        self.report.metadata['completeness'] = completeness_pct
        
        return is_complete
    
    def validate_all(self, df: pd.DataFrame) -> DataQualityReport:
        """Run all validation checks."""
        self.report = DataQualityReport()
        
        # Required columns for UIDAI data
        required_cols = ['date', 'state', 'district', 'age_0_5', 'age_5_17', 'age_18_greater']
        self.validate_schema(df, required_cols)
        
        # Check for missing values
        self.check_missing_values(df)
        
        # Check for duplicates
        self.check_duplicates(df, subset=['date', 'state', 'district'])
        
        # Check date continuity
        self.check_date_continuity(df)
        
        # Check negative values in age columns
        self.check_negative_values(df, ['age_0_5', 'age_5_17', 'age_18_greater'])
        
        # Check value ranges for age columns (reasonable bounds)
        self.check_value_ranges(df, {
            'age_0_5': (0, 10000),
            'age_5_17': (0, 20000),
            'age_18_greater': (0, 50000)
        })
        
        # Check data freshness
        self.check_data_freshness(df)
        
        # Check completeness
        self.check_completeness(df)
        
        # Store metadata
        self.report.metadata.update({
            'total_records': len(df),
            'total_columns': len(df.columns),
            'date_range': {
                'start': str(df['date'].min()) if 'date' in df.columns else None,
                'end': str(df['date'].max()) if 'date' in df.columns else None
            }
        })
        
        return self.report


def generate_quality_report_summary(report: DataQualityReport) -> str:
    """Generate a human-readable summary of the quality report."""
    lines = [
        "=" * 60,
        "DATA QUALITY REPORT",
        "=" * 60,
        f"\nStatus: {'PASSED' if report.is_valid() else 'FAILED'}",
        f"\nChecks: {sum(1 for c in report.checks if c['passed'])}/{len(report.checks)} passed",
        f"Warnings: {len(report.warnings)}",
        f"Errors: {len(report.errors)}",
    ]
    
    if report.errors:
        lines.append("\n" + "-" * 40)
        lines.append("ERRORS:")
        lines.append("-" * 40)
        for error in report.errors:
            lines.append(f"  [!] {error}")
    
    if report.warnings:
        lines.append("\n" + "-" * 40)
        lines.append("WARNINGS:")
        lines.append("-" * 40)
        for warning in report.warnings:
            lines.append(f"  [?] {warning}")
    
    lines.append("\n" + "-" * 40)
    lines.append("DETAILED CHECKS:")
    lines.append("-" * 40)
    for check in report.checks:
        symbol = "[OK]" if check['passed'] else "[!!]"
        lines.append(f"  {symbol} {check['name']}: {check['message']}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test data quality validation
    print("Testing data quality validation...")
    
    # Create sample data with some quality issues
    sample_data = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=10),
        'state': ['Maharashtra'] * 5 + ['Karnataka'] * 5,
        'district': ['Mumbai'] * 5 + ['Bangalore'] * 5,
        'age_0_5': [10, 20, np.nan, 15, 25, 30, 35, 40, 45, 50],
        'age_5_17': [100, 200, 150, 175, 225, 250, 275, 300, 325, 350],
        'age_18_greater': [500, 600, 550, 575, 625, 650, 675, 700, 725, 750]
    })
    
    validator = DataQualityValidator()
    report = validator.validate_all(sample_data)
    
    print(generate_quality_report_summary(report))
