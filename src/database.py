"""
Database module for storing wind forecast results
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://windforecast:windforecast123@localhost:5432/windforecast"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ForecastResult(Base):
    """Table for storing forecast results"""
    __tablename__ = "forecast_results"
    
    id = Column(Integer, primary_key=True, index=True)
    location = Column(String, index=True)
    generated_at = Column(DateTime, default=datetime.now, index=True)
    forecast_horizon_hours = Column(Integer, default=24)
    wind_speeds = Column(JSON)  # Array of 24 hourly predictions
    data_fetched_from = Column(String)  # "Open-Meteo API"
    
    def __repr__(self):
        return f"<Forecast(location={self.location}, generated_at={self.generated_at})>"


class AlertLog(Base):
    """Table for storing alert events"""
    __tablename__ = "alert_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String, index=True)  # HIGH_WIND, RAPID_CHANGE
    severity = Column(String)  # warning, critical
    location = Column(String, index=True)
    detected_at = Column(DateTime, default=datetime.now, index=True)
    threshold = Column(Float)
    actual_value = Column(Float)
    message = Column(String)
    
    def __repr__(self):
        return f"<Alert(type={self.alert_type}, location={self.location}, detected_at={self.detected_at})>"


class ModelMetrics(Base):
    """Table for storing model performance metrics"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    evaluated_at = Column(DateTime, default=datetime.now, index=True)
    mae = Column(Float)
    rmse = Column(Float)
    mape = Column(Float)
    r2_score = Column(Float)
    sample_count = Column(Integer)
    
    def __repr__(self):
        return f"<Metrics(mae={self.mae}, evaluated_at={self.evaluated_at})>"


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # Create tables
    print("Initializing database...")
    init_db()
    
    # Test connection
    from sqlalchemy import text
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print(f"Database connection successful: {result.fetchone()}")
