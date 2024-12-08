# db_utils.py

from sqlalchemy import create_engine
import os

DB_USER = os.environ.get('DB_USER', 'dheeraj')
DB_PASS = os.environ.get('DB_PASS', 'M16226370@d')  # Replace with your actual password
DB_NAME = os.environ.get('DB_NAME', 'krogerdatasetdb')
DB_HOST = os.environ.get('DB_NAME', 'inrtroclouddb.database.windows.net')

engine = create_engine(f"mssql+pyodbc://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?driver=ODBC+Driver+18+for+SQL+Server")
