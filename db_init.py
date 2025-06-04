import mysql.connector
from mysql.connector import Error
import pandas as pd

def connect_to_data():
    db = None
    try:
        db = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            passwd="2431",
            database="reg_finance",
            connect_timeout=60
        )
        cursor = db.cursor()
        return cursor, db
    except Error as e:
        print(f"Ошибка при подключении к базе данных: {e}")
        if db:
            db.close()
        return None, None


def create_df(table):
    cursor, db = connect_to_data()
    if not cursor:
        return None

    try:
        cursor.execute("USE reg_finance")
        cursor.execute(f'SELECT * FROM {table}')
        info = cursor.fetchall()
        column_names = [i[0] for i in cursor.description]
        df = pd.DataFrame(info, columns=column_names)

        if 'код_региона' in df.columns:
            df.set_index('код_региона', inplace=True)
        if 'код_округа' in df.columns:
            df.set_index('код_округа', inplace=True)
        if table in ['коды_регионов', 'коды_округов']:
            df['название'] = df['название'].astype(str)
            return df
        df = convert_df_to_numeric(df)
        return df
    finally:
        if db:
            db.close()

def numeric_convert(value):
    if isinstance(value, str):
        value = value.replace(' ', '')
        if '.' in value:
            value = value.split('.')[0]
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    else:
        return int(value)

def convert_df_to_numeric(df):
    return df.map(numeric_convert)


# Preloaded DataFrames
reg_name = create_df('коды_регионов')
okr_name = create_df('коды_округов')
vrp = create_df('врп')
dkb = create_df('дкб')
rkb = create_df('ркб')
fkp = create_df('фкп')
pkb_r = create_df('пкб_р')
pkb_f = create_df('пкб_фед')
zd_ul = create_df('задолженность_юл')
zd_fl = create_df('задолженность_фл')
vk_ul = create_df('вклады_юл')
vk_fl = create_df('вклады_фл')

table_mapping = {
    'врп': vrp,
    'дкб': dkb,
    'ркб': rkb,
    'пкб_р': pkb_r,
    'пкб_фед': pkb_f,
    'вклады_фл': vk_fl,
    'вклады_юл': vk_ul,
    'задолженность_юл': zd_ul,
    'задолженность_фл': zd_fl,
    'коды_регионов': reg_name,
    'коды_округов': okr_name
}

table_fullname = {
    'врп': " ВАЛОВОЙ РЕГИОНАЛЬНЫЙ ПРОДУКТ",
    'дкб': " ДОХОДЫ КОНСОЛИДИРОВАННЫХ БЮДЖЕТОВ",
    'ркб': " РАСХОДЫ КОНСОЛИДИРОВАННЫХ БЮДЖЕТОВ",
    'пкб_р': " ПОСТУПЛЕНИЕ ОБЯЗАТЕЛЬНЫХ ПЛАТЕЖЕЙ В КОНСОЛИДИРОВАННЫЙ БЮДЖЕТ СУБЪЕКТА",
    'пкб_фед': " ПОСТУПЛЕНИЕ ОБЯЗАТЕЛЬНЫХ ПЛАТЕЖЕЙ В ФЕДЕРАЛЬНЫЙ БЮДЖЕТ",
    'вклады_фл': " ВКЛАДЫ В РУБЛЯХ ФИЗИЧЕСКИХ ЛИЦ",
    'вклады_юл': " ВКЛАДЫ В РУБЛЯХ ЮРИДИЧЕСКИХ ЛИЦ",
    'задолженность_юл': " ЗАДОЛЖЕННОСТЬ В РУБЛЯХ ЮРИДИЧЕСКИХ ЛИЦ",
    'задолженность_фл': " ЗАДОЛЖЕННОСТЬ В РУБЛЯХ ФИЗИЧЕСКИХ ЛИЦ"
}