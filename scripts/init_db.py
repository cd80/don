"""Initialize database tables for Don trading framework."""
import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from don.database.models import Base
from urllib.parse import urlparse

def test_connection(url):
    """Test database connection."""
    try:
        engine = create_engine(url, connect_args={'connect_timeout': 10})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as e:
        print(f"Database connection failed: {str(e)}")
        return False

def main():
    """Create all database tables."""
    load_dotenv()
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("Error: DATABASE_URL not found in environment")
        sys.exit(1)

    print(f"Testing connection to database...")
    if not test_connection(db_url):
        sys.exit(1)

    try:
        engine = create_engine(db_url, connect_args={'connect_timeout': 10})
        Base.metadata.create_all(engine)
        print("Database tables created successfully")
    except SQLAlchemyError as e:
        print(f"Failed to create database tables: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
