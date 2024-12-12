"""Initialize database tables for Don trading framework."""
from don.database.models import Base
from sqlalchemy import create_engine

def main():
    """Create all database tables."""
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/don')
    Base.metadata.create_all(engine)

if __name__ == '__main__':
    main()
