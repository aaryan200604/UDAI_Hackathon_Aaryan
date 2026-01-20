"""
Alert System Module
Threshold-based alerts for UIDAI monitoring.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Callable
from datetime import datetime
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    ENROLMENT_SPIKE = "enrolment_spike"
    ENROLMENT_DROP = "enrolment_drop"
    HIGH_RISK = "high_risk"
    ANOMALY_DETECTED = "anomaly_detected"
    DEMOGRAPHIC_IMBALANCE = "demographic_imbalance"
    VOLATILITY_HIGH = "volatility_high"
    THRESHOLD_BREACH = "threshold_breach"


class Alert:
    """Represents a single alert."""
    
    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        region: str = None,
        value: float = None,
        threshold: float = None,
        timestamp: datetime = None,
        metadata: Dict = None
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.region = region
        self.value = value
        self.threshold = threshold
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            'type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'region': self.region,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def __repr__(self):
        return f"Alert({self.severity.value.upper()}: {self.message})"


class AlertRule:
    """Defines a rule for generating alerts."""
    
    def __init__(
        self,
        name: str,
        alert_type: AlertType,
        column: str,
        condition: Callable,
        severity: AlertSeverity,
        message_template: str,
        threshold: float = None
    ):
        """
        Initialize an alert rule.
        
        Args:
            name: Rule name
            alert_type: Type of alert
            column: Column to check
            condition: Function that takes a value and returns True if alert should fire
            severity: Alert severity
            message_template: Message template (can use {value}, {region}, {threshold})
            threshold: Optional threshold value for reference
        """
        self.name = name
        self.alert_type = alert_type
        self.column = column
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
        self.threshold = threshold


class AlertEngine:
    """Engine for processing and generating alerts."""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.alerts: List[Alert] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Set up default alerting rules."""
        # High risk score alert
        self.add_rule(AlertRule(
            name="high_risk_score",
            alert_type=AlertType.HIGH_RISK,
            column="risk_score",
            condition=lambda x: x > 0.75,
            severity=AlertSeverity.CRITICAL,
            message_template="High risk score detected: {value:.3f} (threshold: {threshold})",
            threshold=0.75
        ))
        
        # Medium risk score alert
        self.add_rule(AlertRule(
            name="medium_risk_score",
            alert_type=AlertType.HIGH_RISK,
            column="risk_score",
            condition=lambda x: 0.5 < x <= 0.75,
            severity=AlertSeverity.WARNING,
            message_template="Elevated risk score: {value:.3f}",
            threshold=0.5
        ))
        
        # Anomaly detection alert
        self.add_rule(AlertRule(
            name="anomaly_detected",
            alert_type=AlertType.ANOMALY_DETECTED,
            column="is_anomaly",
            condition=lambda x: x == True,
            severity=AlertSeverity.WARNING,
            message_template="Anomaly detected in enrolment patterns"
        ))
        
        # Demographic imbalance alert
        self.add_rule(AlertRule(
            name="demographic_imbalance",
            alert_type=AlertType.DEMOGRAPHIC_IMBALANCE,
            column="demographic_imbalance",
            condition=lambda x: x > 0.2,
            severity=AlertSeverity.WARNING,
            message_template="Demographic imbalance detected: {value:.2%} deviation",
            threshold=0.2
        ))
        
        # High volatility alert
        self.add_rule(AlertRule(
            name="high_volatility",
            alert_type=AlertType.VOLATILITY_HIGH,
            column="volatility_cv",
            condition=lambda x: x > 0.5,
            severity=AlertSeverity.WARNING,
            message_template="High enrolment volatility: CV = {value:.2f}",
            threshold=0.5
        ))
        
        # Enrolment spike alert
        self.add_rule(AlertRule(
            name="enrolment_spike",
            alert_type=AlertType.ENROLMENT_SPIKE,
            column="growth_rate",
            condition=lambda x: x > 0.5,
            severity=AlertSeverity.WARNING,
            message_template="Enrolment spike: {value:.1%} increase",
            threshold=0.5
        ))
        
        # Enrolment drop alert
        self.add_rule(AlertRule(
            name="enrolment_drop",
            alert_type=AlertType.ENROLMENT_DROP,
            column="growth_rate",
            condition=lambda x: x < -0.3,
            severity=AlertSeverity.WARNING,
            message_template="Enrolment drop: {value:.1%} decrease",
            threshold=-0.3
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules.append(rule)
    
    def remove_rule(self, name: str):
        """Remove a rule by name."""
        self.rules = [r for r in self.rules if r.name != name]
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []
    
    def process(
        self,
        df: pd.DataFrame,
        region_col: str = 'state',
        date_col: str = 'date'
    ) -> List[Alert]:
        """
        Process DataFrame and generate alerts based on rules.
        
        Args:
            df: DataFrame to process
            region_col: Column containing region names
            date_col: Column containing dates
            
        Returns:
            List of generated alerts
        """
        self.clear_alerts()
        
        for _, row in df.iterrows():
            for rule in self.rules:
                if rule.column not in df.columns:
                    continue
                
                value = row[rule.column]
                
                # Skip NaN values
                if pd.isna(value):
                    continue
                
                # Check condition
                try:
                    if rule.condition(value):
                        # Generate alert
                        region = row.get(region_col, 'Unknown')
                        timestamp = row.get(date_col, datetime.now())
                        
                        message = rule.message_template.format(
                            value=value,
                            region=region,
                            threshold=rule.threshold
                        )
                        
                        alert = Alert(
                            alert_type=rule.alert_type,
                            severity=rule.severity,
                            message=message,
                            region=region,
                            value=value,
                            threshold=rule.threshold,
                            timestamp=timestamp if isinstance(timestamp, datetime) else datetime.now(),
                            metadata={'rule': rule.name}
                        )
                        
                        self.alerts.append(alert)
                except Exception:
                    pass
        
        return self.alerts
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts filtered by severity."""
        return [a for a in self.alerts if a.severity == severity]
    
    def get_alerts_by_type(self, alert_type: AlertType) -> List[Alert]:
        """Get alerts filtered by type."""
        return [a for a in self.alerts if a.alert_type == alert_type]
    
    def get_alerts_by_region(self, region: str) -> List[Alert]:
        """Get alerts filtered by region."""
        return [a for a in self.alerts if a.region == region]
    
    def get_summary(self) -> Dict:
        """Get summary of all alerts."""
        return {
            'total_alerts': len(self.alerts),
            'critical': len(self.get_alerts_by_severity(AlertSeverity.CRITICAL)),
            'warnings': len(self.get_alerts_by_severity(AlertSeverity.WARNING)),
            'info': len(self.get_alerts_by_severity(AlertSeverity.INFO)),
            'by_type': {
                at.value: len(self.get_alerts_by_type(at))
                for at in AlertType
            },
            'affected_regions': list(set(a.region for a in self.alerts if a.region))
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert alerts to DataFrame."""
        return pd.DataFrame([a.to_dict() for a in self.alerts])


def configure_thresholds(
    risk_high: float = 0.75,
    risk_medium: float = 0.5,
    imbalance: float = 0.2,
    volatility: float = 0.5,
    growth_spike: float = 0.5,
    growth_drop: float = -0.3
) -> AlertEngine:
    """
    Create an AlertEngine with custom thresholds.
    
    Args:
        risk_high: Threshold for high risk alerts
        risk_medium: Threshold for medium risk alerts
        imbalance: Threshold for demographic imbalance
        volatility: Threshold for volatility alerts
        growth_spike: Threshold for growth spike alerts
        growth_drop: Threshold for growth drop alerts
        
    Returns:
        Configured AlertEngine
    """
    engine = AlertEngine()
    engine.rules = []  # Clear defaults
    
    # Re-add with custom thresholds
    engine.add_rule(AlertRule(
        name="high_risk_score",
        alert_type=AlertType.HIGH_RISK,
        column="risk_score",
        condition=lambda x, t=risk_high: x > t,
        severity=AlertSeverity.CRITICAL,
        message_template="High risk score detected: {value:.3f}",
        threshold=risk_high
    ))
    
    engine.add_rule(AlertRule(
        name="medium_risk_score",
        alert_type=AlertType.HIGH_RISK,
        column="risk_score",
        condition=lambda x, t1=risk_medium, t2=risk_high: t1 < x <= t2,
        severity=AlertSeverity.WARNING,
        message_template="Elevated risk score: {value:.3f}",
        threshold=risk_medium
    ))
    
    engine.add_rule(AlertRule(
        name="demographic_imbalance",
        alert_type=AlertType.DEMOGRAPHIC_IMBALANCE,
        column="demographic_imbalance",
        condition=lambda x, t=imbalance: x > t,
        severity=AlertSeverity.WARNING,
        message_template="Demographic imbalance: {value:.2%}",
        threshold=imbalance
    ))
    
    engine.add_rule(AlertRule(
        name="high_volatility",
        alert_type=AlertType.VOLATILITY_HIGH,
        column="volatility_cv",
        condition=lambda x, t=volatility: x > t,
        severity=AlertSeverity.WARNING,
        message_template="High volatility: CV = {value:.2f}",
        threshold=volatility
    ))
    
    engine.add_rule(AlertRule(
        name="enrolment_spike",
        alert_type=AlertType.ENROLMENT_SPIKE,
        column="growth_rate",
        condition=lambda x, t=growth_spike: x > t,
        severity=AlertSeverity.WARNING,
        message_template="Enrolment spike: {value:.1%}",
        threshold=growth_spike
    ))
    
    engine.add_rule(AlertRule(
        name="enrolment_drop",
        alert_type=AlertType.ENROLMENT_DROP,
        column="growth_rate",
        condition=lambda x, t=growth_drop: x < t,
        severity=AlertSeverity.WARNING,
        message_template="Enrolment drop: {value:.1%}",
        threshold=growth_drop
    ))
    
    return engine


if __name__ == "__main__":
    # Test alert system
    print("Testing alert system...")
    
    sample_data = pd.DataFrame({
        'state': ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'UP', 'Gujarat'],
        'risk_score': [0.85, 0.45, 0.30, 0.78, 0.55],
        'is_anomaly': [True, False, False, True, False],
        'demographic_imbalance': [0.25, 0.10, 0.05, 0.22, 0.08],
        'volatility_cv': [0.65, 0.30, 0.20, 0.55, 0.40],
        'growth_rate': [0.60, 0.10, -0.35, 0.15, 0.08]
    })
    
    engine = AlertEngine()
    alerts = engine.process(sample_data)
    
    print(f"\nGenerated {len(alerts)} alerts:")
    for alert in alerts:
        print(f"  [{alert.severity.value.upper()}] {alert.region}: {alert.message}")
    
    print("\nSummary:")
    summary = engine.get_summary()
    print(f"  Critical: {summary['critical']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Affected regions: {summary['affected_regions']}")
