from mysql.connector import Error
import pandas as pd
import db_init

def delete_data(region_code):
    cursor, db = db_init.connect_to_data()
    if not cursor:
        return

    try:
        table_name = input("📁 Введите имя таблицы для удаления данных: ").strip()
        if table_name not in db_init.table_mapping:
            print(f"❗ Таблица '{table_name}' не найдена.")
            return

        confirmation = input(f"⚠️ Удалить все данные для региона {region_code} из таблицы '{table_name}'? (y/n): ").strip().lower()
        if confirmation == 'y':
            cursor.execute("USE reg_finance")
            cursor.execute(f"DELETE FROM `{table_name}` WHERE код_региона = %s", (region_code,))
            db.commit()
            print(f"✅ Данные региона {region_code} удалены из '{table_name}'.")
        else:
            print("❎ Удаление отменено.")
    except Error as e:
        print(f"💥 Ошибка при удалении данных: {e}")
        db.rollback()
    finally:
        if db:
            db.close()

def add_data(region_code):
    cursor, db = db_init.connect_to_data()
    if not cursor:
        return

    try:
        table_name = input("📁 Введите имя таблицы для добавления данных: ").strip()
        if table_name not in db_init.table_mapping:
            print(f"❗ Таблица '{table_name}' не найдена.")
            return

        df = db_init.create_df(table_name)
        if df is None:
            print("⚠️ Не удалось загрузить таблицу.")
            return

        column_names = list(df.columns)
        if 'id' in column_names:
            column_names.remove('id')
        if 'код_региона' in column_names:
            column_names.remove('код_региона')

        values = [db_init.numeric_convert(region_code)]
        columns_for_query = ['код_региона']

        print("🖊️ Введите значения для столбцов (оставьте пустым для NULL):")
        for col in column_names:
            value = input(f"  ➤ {col}: ").strip()
            values.append(None if value == "" else db_init.numeric_convert(value))
            columns_for_query.append(col)

        placeholders = ', '.join(['%s'] * len(columns_for_query))
        escaped_columns = [f"`{column}`" for column in columns_for_query]
        sql = f"INSERT INTO `{table_name}` ({', '.join(escaped_columns)}) VALUES ({placeholders})"

        cursor.execute("USE reg_finance")
        cursor.execute(sql, values)
        db.commit()
        print("✅ Данные успешно добавлены.")
    except Error as e:
        print(f"💥 Ошибка при добавлении данных: {e}")
        db.rollback()
    finally:
        if db:
            db.close()

def edit_data(region_code):
    cursor, db = db_init.connect_to_data()
    if not cursor:
        return

    try:
        table_name = input("📁 Введите имя таблицы для редактирования: ").strip()
        if table_name not in db_init.table_mapping:
            print(f"❗ Таблица '{table_name}' не найдена.")
            return

        df = db_init.create_df(table_name)
        if df is None or df.empty:
            print("⚠️ Таблица пуста или не загружена.")
            return
        if int(region_code) not in df.index:
            print(f"❗ Регион {region_code} отсутствует в таблице.")
            return

        print("🛠️ Доступные столбцы для редактирования:")
        for i, col in enumerate(df.columns):
            print(f"  {i + 1}. {col}")
        column_index = int(input("👉 Выберите номер столбца: ")) - 1
        column_name = df.columns[column_index]

        new_value = input(f"🖊️ Новое значение для '{column_name}': ").strip()
        new_value = None if new_value == "" else db_init.numeric_convert(new_value)

        sql = f"UPDATE `{table_name}` SET `{column_name}` = %s WHERE код_региона = %s"
        cursor.execute("USE reg_finance")
        cursor.execute(sql, (new_value, region_code))
        db.commit()
        print("✅ Данные успешно обновлены.")
    except Error as e:
        print(f"💥 Ошибка при редактировании данных: {e}")
        db.rollback()
    finally:
        if db:
            db.close()

def view_data():
    table_name = input("📁 Введите имя таблицы для просмотра: ").strip()
    if table_name not in db_init.table_mapping:
        print(f"❗ Таблица '{table_name}' не найдена.")
        return
    df = db_init.create_df(table_name)
    if df is None or df.empty:
        print("⚠️ Таблица пуста или не загружена.")
        return
    print(f"\n📄 Содержимое таблицы '{table_name}':")
    print(df)

def show_region_data(region_code):
    cursor, db = db_init.connect_to_data()
    if not cursor:
        return

    try:
        cursor.execute("USE reg_finance")
        excluded_tables = {'коды_регионов', 'коды_округов'}
        for table in db_init.table_mapping:
            if table in excluded_tables:
                continue
            cursor.execute(f"SELECT * FROM `{table}` WHERE код_региона = %s", (region_code,))
            rows = cursor.fetchall()
            if rows:
                print(f"\n📂 Таблица: {table}")
                for row in rows:
                    print(row)
    except Error as e:
        print(f"💥 Ошибка при получении данных: {e}")
    finally:
        if db:
            db.close()


def show_multiple_regions_data(table_name, region_codes):
    cursor, db = db_init.connect_to_data()
    if not cursor:
        return

    try:
        cursor.execute("USE reg_finance")
        placeholders = ', '.join(['%s'] * len(region_codes))
        sql = f"SELECT * FROM `{table_name}` WHERE код_региона IN ({placeholders})"
        cursor.execute(sql, tuple(region_codes))
        rows = cursor.fetchall()
        if rows:
            print(f"\n📂 Таблица: {table_name}")
            for row in rows:
                print(row)
        else:
            print("ℹ️ Нет данных для указанных регионов.")
    except Error as e:
        print(f"💥 Ошибка при получении данных: {e}")
    finally:
        if db:
            db.close()
