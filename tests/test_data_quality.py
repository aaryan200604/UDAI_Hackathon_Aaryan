"""
Unit Tests for Data Quality Module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.quality import DataQualityValidator, DataQualityReport, generate_quality_report_summary


class TestDataQualityReport:
    """Test DataQualityReport class."""
    
    def test_report_initialization(self):
        """Test report initialization."""
        report = DataQualityReport()
        assert len(report.checks) == 0
        assert len(report.warnings) == 0
        assert len(report.errors) == 0
        assert report.is_valid()
    
    def test_add_check(self):
        """Test adding checks."""
        report = DataQualityReport()
        report.add_check('test_check', True, 'Check passed')
        
        assert len(report.checks) == 1
        assert report.is_valid()
        
        report.add_check('failed_check', False, 'Check failed')
        assert not report.is_valid()
        assert len(report.errors) == 1
    
    def test_add_warning(self):
        """Test adding warnings."""
        report = DataQualityReport()
        report.add_warning('Test warning')
        
        assert len(report.warnings) == 1
        assert report.is_valid()  # Warnings don't affect validity
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = DataQualityReport()
        report.add_check('check1', True, 'Passed')
        report.add_check('check2', False, 'Failed')
        report.add_warning('Warning')
        
        result = report.to_dict()
        
        assert result['total_checks'] == 2
        assert result['passed'] == 1
        assert result['failed'] == 1
        assert result['warnings'] == 1
        assert not result['valid']


class TestDataQualityValidator:
    """Test DataQualityValidator class."""
    
    def setup_method(self):
        """Set up test data."""
        self.valid_df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=10),
            'state': ['State A'] * 5 + ['State B'] * 5,
            'district': ['D1'] * 5 + ['D2'] * 5,
            'age_0_5': [10, 15, 20, 25, 30, 12, 18, 22, 28, 32],
            'age_5_17': [50, 55, 60, 65, 70, 52, 58, 62, 68, 72],
            'age_18_greater': [200, 210, 220, 230, 240, 205, 215, 225, 235, 245]
        })
    
    def test_validate_schema_valid(self):
        """Test schema validation with valid data."""
        validator = DataQualityValidator()
        required_cols = ['date', 'state', 'district']
        
        result = validator.validate_schema(self.valid_df, required_cols)
        assert result is True
    
    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing columns."""
        validator = DataQualityValidator()
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        result = validator.validate_schema(df, ['col1', 'col2', 'col3'])
        assert result is False
    
    def test_check_missing_values(self):
        """Test missing value check."""
        validator = DataQualityValidator()
        df = self.valid_df.copy()
        df.loc[0:2, 'age_0_5'] = np.nan  # 3 out of 10 = 30% missing
        
        # 30% is above 20% threshold, so check should FAIL (return False)
        result = validator.check_missing_values(df, threshold=0.2)
        assert result is False
        
        # 30% is below 40% threshold, so check should PASS (return True)
        result = validator.check_missing_values(df, threshold=0.4)
        assert result is True
    
    def test_check_duplicates(self):
        """Test duplicate check."""
        validator = DataQualityValidator()
        
        # No duplicates
        result = validator.check_duplicates(self.valid_df)
        assert result is True
        
        # With duplicates
        df_with_dupes = pd.concat([self.valid_df, self.valid_df.head(2)])
        result = validator.check_duplicates(df_with_dupes)
        assert result is False
    
    def test_check_date_continuity(self):
        """Test date continuity check."""
        validator = DataQualityValidator()
        
        # Continuous dates
        result = validator.check_date_continuity(self.valid_df)
        assert result is True
        
        # With gaps
        df_with_gaps = self.valid_df[self.valid_df['date'] != pd.Timestamp('2025-01-05')]
        result = validator.check_date_continuity(df_with_gaps)
        assert result is False
    
    def test_check_negative_values(self):
        """Test negative value check."""
        validator = DataQualityValidator()
        
        # No negatives
        result = validator.check_negative_values(self.valid_df, ['age_0_5', 'age_5_17'])
        assert result is True
        
        # With negatives
        df = self.valid_df.copy()
        df.loc[0, 'age_0_5'] = -5
        result = validator.check_negative_values(df, ['age_0_5'])
        assert result is False
    
    def test_check_value_ranges(self):
        """Test value range check."""
        validator = DataQualityValidator()
        
        ranges = {
            'age_0_5': (0, 100),
            'age_5_17': (0, 200)
        }
        
        result = validator.check_value_ranges(self.valid_df, ranges)
        assert result is True
        
        # Out of range
        df = self.valid_df.copy()
        df.loc[0, 'age_0_5'] = 150
        result = validator.check_value_ranges(df, ranges)
        assert result is False
    
    def test_validate_all(self):
        """Test complete validation."""
        validator = DataQualityValidator()
        report = validator.validate_all(self.valid_df)
        
        assert isinstance(report, DataQualityReport)
        assert len(report.checks) > 0
        assert 'total_records' in report.metadata
        assert 'completeness' in report.metadata


class TestQualityReportSummary:
    """Test quality report summary generation."""
    
    def test_generate_summary(self):
        """Test summary generation."""
        report = DataQualityReport()
        report.add_check('check1', True, 'Passed')
        report.add_check('check2', False, 'Failed')
        report.add_warning('Test warning')
        
        summary = generate_quality_report_summary(report)
        
        assert isinstance(summary, str)
        assert 'FAILED' in summary
        assert 'Test warning' in summary
        assert '1/2 passed' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
