# main.py

import os
from flask import Flask, render_template, request, redirect, url_for, session
from sqlalchemy import text
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')  # Replace with a strong secret key in production

# Import the SQLAlchemy engine from db_utils.py
from db_utils import engine

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    """
    Handle user login. Assume any provided credentials are valid.
    """
    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email')  # Keep if used

    if username and password and email:
        session['user'] = username
        logger.info(f"User '{username}' logged in successfully with email '{email}'.")
        return redirect(url_for('search_page'))
    else:
        logger.warning("Login attempt with missing credentials.")
        return "Please provide username, password, and email.", 400

@app.route('/logout')
def logout():
    """
    Handle user logout by clearing the session.
    """
    user = session.pop('user', None)
    if user:
        logger.info(f"User '{user}' logged out.")
    return redirect(url_for('index'))

@app.route('/search', methods=['GET', 'POST'])
def search_page():
    """
    Handle search functionality for household numbers.
    """
    if 'user' not in session:
        logger.warning("Unauthorized access attempt to search page.")
        return redirect(url_for('index'))
    
    results = None
    if request.method == 'POST':
        hshd_num = request.form.get('hshd_num')
        if hshd_num:
            try:
                hshd_num_int = int(hshd_num)  # Ensure it's an integer
                query = text("""
                    SELECT h.HSHD_NUM, t.BASKET_NUM, t.DATE, t.PRODUCT_NUM, p.DEPARTMENT, 
                           p.COMMODITY, t.SPEND, t.UNITS, t.STORE_REGION, t.WEEK_NUM, t.YEAR
                    FROM Transactions t
                    JOIN Products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                    JOIN Households h ON t.HSHD_NUM = h.HSHD_NUM
                    WHERE h.HSHD_NUM = :hshd_num
                    ORDER BY h.HSHD_NUM, t.BASKET_NUM, t.DATE, t.PRODUCT_NUM, p.DEPARTMENT, p.COMMODITY
                """)
                with engine.connect() as conn:
                    results = conn.execute(query, hshd_num=hshd_num_int).fetchall()
                    logger.info(f"Fetched {len(results)} records for HSHD_NUM={hshd_num_int}")
            except ValueError:
                logger.error(f"Invalid HSHD_NUM input: {hshd_num}")
                return "Invalid Household Number.", 400
            except Exception as e:
                logger.error(f"Database query failed for HSHD_NUM={hshd_num}: {e}")
                return "An error occurred while fetching data.", 500
    return render_template('search.html', results=results)

@app.route('/dashboard')
def dashboard_page():
    """
    Display the dashboard with static data.
    """
    if 'user' not in session:
        logger.warning("Unauthorized access attempt to dashboard.")
        return redirect(url_for('index'))

    # Since dashboard.html now contains static data, simply render the template without passing any variables.
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
