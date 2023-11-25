from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    dp_url = Column(String, nullable=False)


# Specify the database driver in the connection string (mysql+mysqlconnector)
engine = create_engine("mysql+mysqlconnector://root:admin@localhost/mydb")

# Create the tables within a try-except block to catch potential exceptions
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"An error occurred while creating database tables: {e}")

# Set up a session
Session = sessionmaker(bind=engine)
session = Session()
