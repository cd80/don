"""Initialize database tables for Don trading framework."""
import os
from dotenv import load_dotenv
from don.database.models import Base
from sqlalchemy import create_engine

def main():
    """Create all database tables."""
    load_dotenv()
    engine = create_engine(os.getenv('DATABASE_URL'))
    Base.metadata.create_all(engine)

if __name__ == '__main__':
    main()
