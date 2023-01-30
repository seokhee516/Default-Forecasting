import psycopg2
import pickle
import pandas as pd
import time


def get_data():
    # Raw data 읽기
    df = pd.read_csv("/opt/ml/loan/LendingClub_Loan_status_2018-2020Q3.csv").iloc[:, 1:]
    # 19년 ~ 20년 데이터 사용
    sepc = df["issue_d"] >= "Sep-2019"
    octc = (df["issue_d"] >= "Oct-2019") & (df["issue_d"] <= "Oct-2020")
    novc = (df["issue_d"] >= "Nov-2019") & (df["issue_d"] <= "Nov-2020")
    mayc = (df["issue_d"] >= "May-2019") & (df["issue_d"] <= "May-2020")
    df = df[sepc | octc | novc | mayc].reset_index().iloc[:, 2:]
    # loan_status 변수를 정상(0)과 불량(1)으로 이진 분류하여 Target 변수 생성
    df.loc[
        (df["loan_status"] == "Fully Paid") | (df["loan_status"] == "Current"), "Target"
    ] = 0  # 정상(0)
    df.loc[
        (df["loan_status"] == "Late (31-120 days)")
        | (df["loan_status"] == "Charged Off")
        | (df["loan_status"] == "In Grace Period")
        | (df["loan_status"] == "Late (16-30 days)")
        | (df["loan_status"] == "Default")
        | (df["loan_status"] == "Issued")
        | (df["loan_status"] == "Late (31-120 days)"),
        "Target",
    ] = 1  # 불량(1)
    # 141 특성 중 30개만 사용
    columns = [
        "annual_inc",
        "inq_last_6mths",
        "home_ownership",
        "purpose",
        "last_fico_range_high",
        "last_fico_range_low",
        "sub_grade",
        "int_rate",
        "installment",
        "tot_cur_bal",
        "avg_cur_bal",
        "mo_sin_old_rev_tl_op",
        "mo_sin_rcnt_rev_tl_op",
        "mo_sin_rcnt_tl",
        "mort_acc",
        "num_il_tl",
        "emp_length",
        "num_tl_op_past_12m",
        "revol_bal",
        "total_bc_limit",
        "dti",
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "tot_hi_cred_lim",
        "Target",
        "issue_d",
    ]  # id, Target, issue_d: Date
    df = df[columns]
    df = df.dropna()
    df.reset_index(inplace=True)
    df.rename(columns={"index": "id"}, inplace=True)
    return df


def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS lending (
        id SERIAL PRIMARY KEY,
        annual_inc FLOAT,
        inq_last_6mths FLOAT,
        home_ownership VARCHAR(128),
        purpose VARCHAR(128),
        last_fico_range_high FLOAT,
        last_fico_range_low FLOAT,
        sub_grade VARCHAR(128),
        int_rate VARCHAR(128),
        installment FLOAT,
        tot_cur_bal FLOAT,
        avg_cur_bal FLOAT,
        mo_sin_old_rev_tl_op FLOAT,
        mo_sin_rcnt_rev_tl_op FLOAT,
        mo_sin_rcnt_tl FLOAT,
        mort_acc FLOAT,
        num_il_tl FLOAT,
        emp_length VARCHAR(128),
        num_tl_op_past_12m FLOAT,
        revol_bal FLOAT,
        total_bc_limit FLOAT,
        dti FLOAT,
        out_prncp FLOAT,
        out_prncp_inv FLOAT,
        total_pymnt FLOAT,
        total_pymnt_inv FLOAT,
        total_rec_prncp FLOAT,
        total_rec_int FLOAT,
        total_rec_late_fee FLOAT,
        tot_hi_cred_lim FLOAT,
        Target FLOAT,
        issue_d VARCHAR(128));
        """
    print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query.format(42))
        db_connect.commit()


def insert_data(db_connect, data):
    insert_row_query = """ INSERT INTO lending (annual_inc, inq_last_6mths, home_ownership, purpose, \
        last_fico_range_high, last_fico_range_low, sub_grade, int_rate, \
        installment, tot_cur_bal, avg_cur_bal, mo_sin_old_rev_tl_op, \
        mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl, mort_acc, num_il_tl, \
        emp_length, num_tl_op_past_12m, revol_bal, total_bc_limit, \
        dti, out_prncp, out_prncp_inv, total_pymnt, total_pymnt_inv, \
        total_rec_prncp, total_rec_int, total_rec_late_fee, \
        tot_hi_cred_lim, Target, issue_d) \
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"""

    print(insert_row_query)
    with db_connect.cursor() as cur:
        cur.execute(
            insert_row_query,
            (
                data.annual_inc,
                data.inq_last_6mths,
                data.home_ownership,
                data.purpose,
                data.last_fico_range_high,
                data.last_fico_range_low,
                data.sub_grade,
                data.int_rate,
                data.installment,
                data.tot_cur_bal,
                data.avg_cur_bal,
                data.mo_sin_old_rev_tl_op,
                data.mo_sin_rcnt_rev_tl_op,
                data.mo_sin_rcnt_tl,
                data.mort_acc,
                data.num_il_tl,
                data.emp_length,
                data.num_tl_op_past_12m,
                data.revol_bal,
                data.total_bc_limit,
                data.dti,
                data.out_prncp,
                data.out_prncp_inv,
                data.total_pymnt,
                data.total_pymnt_inv,
                data.total_rec_prncp,
                data.total_rec_int,
                data.total_rec_late_fee,
                data.tot_hi_cred_lim,
                data.Target,
                data.issue_d,
            ),
        )
        db_connect.commit()


def generate_data(db_connect, df):
    while True:
        insert_data(db_connect, df.sample(1).squeeze())
        time.sleep(1)


if __name__ == "__main__":
    with open("secret_key.p", "rb") as file:  # secret_key 읽기
        secret_key = pickle.load(file)
        db_connect = psycopg2.connect(
            host=secret_key["host"],
            database=secret_key["database"],
            user=secret_key["user"],
            password=secret_key["password"],
        )
        df = get_data()
        create_table(db_connect)
        generate_data(db_connect, df)