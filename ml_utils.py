# ml_utils.py

import pandas as pd
from sqlalchemy import text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from db_utils import engine
import warnings
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def basket_analysis():
    """
    Perform market basket analysis to find frequently co-purchased product pairs.
    """
    query = text("""
        SELECT t1.PRODUCT_NUM AS product1, t2.PRODUCT_NUM AS product2, COUNT(*) AS frequency
        FROM Transactions t1
        JOIN Transactions t2 
          ON t1.BASKET_NUM = t2.BASKET_NUM
         AND t1.PRODUCT_NUM < t2.PRODUCT_NUM
        GROUP BY t1.PRODUCT_NUM, t2.PRODUCT_NUM
        HAVING frequency > 10
        ORDER BY frequency DESC
        LIMIT 10
    """)
    try:
        logger.info("Executing basket_analysis query...")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        logger.info("Basket Analysis: Successfully fetched top 10 product pairs.")
        return df
    except Exception as e:
        logger.error(f"Basket Analysis failed: {e}")
        return None


def churn_prediction():
    """
    Predict customer churn using Random Forest.
    """
    warnings.filterwarnings("ignore")
    query = text("""
        SELECT h.HSHD_NUM,
               MAX(t.DATE) AS last_purchase,
               COUNT(DISTINCT t.BASKET_NUM) AS total_baskets,
               SUM(t.SPEND) AS total_spend,
               AVG(t.SPEND) AS avg_spend
        FROM Households h
        JOIN Transactions t ON h.HSHD_NUM = t.HSHD_NUM
        GROUP BY h.HSHD_NUM
    """)
    try:
        logger.info("Executing churn_prediction query...")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            logger.warning("Churn Prediction: No data available.")
            return "N/A", "N/A"

        df['last_purchase'] = pd.to_datetime(df['last_purchase'])
        last_date = df['last_purchase'].max()
        df['days_since_purchase'] = (last_date - df['last_purchase']).dt.days
        df['churned'] = (df['days_since_purchase'] > 90).astype(int)

        features = ['total_baskets', 'total_spend', 'avg_spend', 'days_since_purchase']
        X = df[features].fillna(0)
        y = df['churned']

        if y.nunique() < 2:
            logger.warning("Churn Prediction: Not enough variation in target variable.")
            return "N/A", "N/A"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        churn_rate = y.mean()

        logger.info(f"Churn Prediction: Model Accuracy = {score * 100:.2f}%, Churn Rate = {churn_rate * 100:.2f}%")
        return f"{score * 100:.2f}", f"{churn_rate * 100:.2f}"

    except Exception as e:
        logger.error(f"Churn Prediction failed: {e}")
        return "N/A", "N/A"


def clv_prediction():
    """
    Predict Customer Lifetime Value (CLV) using Gradient Boosting.
    """
    warnings.filterwarnings("ignore")
    query = text("""
        SELECT h.HSHD_NUM,
               COUNT(DISTINCT t.BASKET_NUM) AS total_baskets,
               SUM(t.SPEND) AS total_spend,
               AVG(t.SPEND) AS avg_spend,
               MAX(t.DATE) AS last_purchase,
               h.HSHD_SIZE,
               h.CHILDREN,
               h.INCOME_RANGE
        FROM Households h
        JOIN Transactions t ON h.HSHD_NUM = t.HSHD_NUM
        GROUP BY h.HSHD_NUM, h.HSHD_SIZE, h.CHILDREN, h.INCOME_RANGE
    """)
    try:
        logger.info("Executing clv_prediction query...")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            logger.warning("CLV Prediction: No data available.")
            return "N/A", "N/A", None

        df['last_purchase'] = pd.to_datetime(df['last_purchase'])
        df['days_since_purchase'] = (df['last_purchase'].max() - df['last_purchase']).dt.days

        df['CLV'] = df['total_spend']

        df = pd.get_dummies(df, columns=['HSHD_SIZE', 'CHILDREN', 'INCOME_RANGE'], drop_first=True)

        features = [col for col in df.columns if col not in ['HSHD_NUM', 'CLV', 'last_purchase']]
        X = df[features]
        y = df['CLV']

        if X.empty or y.empty:
            logger.warning("CLV Prediction: Features or target variable is empty.")
            return "N/A", "N/A", None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"CLV Prediction: R-squared = {r2:.2f}, Mean Absolute Error = {mae:.2f}")
        return f"{r2:.2f}", f"{mae:.2f}", model

    except Exception as e:
        logger.error(f"CLV Prediction failed: {e}")
        return "N/A", "N/A", None
