"""
初始化本地模拟金融数据库 (SQLite)
生成带有统计学显著规律的客户资产 + 行为数据，供 Agent 的 Text-to-SQL 工具查询。
"""
import os
import sqlite3
import numpy as np
import pandas as pd


def initialize_database():
    db_path = os.path.join("data", "finance.db")
    os.makedirs("data", exist_ok=True)

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("正在构建模拟金融数据库...")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_profile (
            customer_id TEXT PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            account_balance REAL,
            credit_score INTEGER,
            has_churned INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_behavior (
            customer_id TEXT,
            transaction_count_3m INTEGER,
            complaint_count INTEGER,
            last_login_days INTEGER,
            FOREIGN KEY (customer_id) REFERENCES customer_profile(customer_id)
        )
    """)

    # 新增：贷款表，让 schema 更丰富，方便测试多表 JOIN 的 Text-to-SQL 能力
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS loan_records (
            loan_id TEXT PRIMARY KEY,
            customer_id TEXT,
            loan_amount REAL,
            interest_rate REAL,
            is_default INTEGER,
            FOREIGN KEY (customer_id) REFERENCES customer_profile(customer_id)
        )
    """)

    np.random.seed(42)
    n_samples = 2000

    c_ids = [f"CUST_{i:04d}" for i in range(n_samples)]
    ages = np.random.randint(18, 75, size=n_samples)
    genders = np.random.choice(["Male", "Female"], size=n_samples)

    balances = np.random.exponential(scale=50000, size=n_samples) + 500
    credit_scores = np.clip(np.random.normal(loc=650, scale=80, size=n_samples).astype(int), 300, 850)

    churn_prob = 1 / (1 + np.exp(-(-0.00002 * balances - 0.005 * credit_scores
                                    + 0.8 * np.random.randint(0, 5, size=n_samples) + 2)))
    has_churned = np.random.binomial(1, p=churn_prob)

    profile_df = pd.DataFrame({
        "customer_id": c_ids,
        "age": ages,
        "gender": genders,
        "account_balance": np.round(balances, 2),
        "credit_score": credit_scores,
        "has_churned": has_churned,
    })
    profile_df.to_sql("customer_profile", conn, if_exists="append", index=False)

    tx_counts = np.random.poisson(lam=15, size=n_samples)
    tx_counts = np.where(has_churned == 1, (tx_counts * 0.4).astype(int), tx_counts)
    complaints = np.random.poisson(lam=0.2, size=n_samples)
    complaints = np.where(has_churned == 1, complaints + np.random.randint(1, 3, size=n_samples), complaints)
    logins = np.random.randint(1, 30, size=n_samples)
    logins = np.where(has_churned == 1, logins * 2, logins)

    behavior_df = pd.DataFrame({
        "customer_id": c_ids,
        "transaction_count_3m": tx_counts,
        "complaint_count": complaints,
        "last_login_days": logins,
    })
    behavior_df.to_sql("customer_behavior", conn, if_exists="append", index=False)

    # 贷款数据：约 60% 的客户有贷款记录，低信用分客户违约率显著更高
    has_loan_mask = np.random.rand(n_samples) < 0.6
    loan_customers = np.array(c_ids)[has_loan_mask]
    loan_credit_scores = credit_scores[has_loan_mask]

    n_loans = len(loan_customers)
    loan_ids = [f"LOAN_{i:04d}" for i in range(n_loans)]
    loan_amounts = np.round(np.random.exponential(scale=20000, size=n_loans) + 1000, 2)
    interest_rates = np.round(np.random.uniform(3.5, 18.0, size=n_loans), 2)

    default_prob = 1 / (1 + np.exp(-(-0.01 * loan_credit_scores + 5)))
    is_default = np.random.binomial(1, p=default_prob)

    loan_df = pd.DataFrame({
        "loan_id": loan_ids,
        "customer_id": loan_customers,
        "loan_amount": loan_amounts,
        "interest_rate": interest_rates,
        "is_default": is_default,
    })
    loan_df.to_sql("loan_records", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    print(f"数据库初始化成功！文件保存在: {db_path}")
    print(f"  - customer_profile: {n_samples} 条")
    print(f"  - customer_behavior: {n_samples} 条")
    print(f"  - loan_records: {n_loans} 条")


if __name__ == "__main__":
    initialize_database()
