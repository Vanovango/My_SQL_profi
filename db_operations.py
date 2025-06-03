from mysql.connector import Error
import pandas as pd
import db_init

def delete_data():
    mycursor, db = db_init.connect_to_data()
    if not mycursor:
        return

    try:
        table_name = input("Введите имя таблицы, из которой хотите удалить данные: ").strip()
        if table_name not in db_init.table_mapping:
            print(f"Таблица '{table_name}' не найдена.")
            return

        confirmation = input(f"Вы уверены, что хотите УДАЛИТЬ ВСЕ данные из таблицы '{table_name}'? (y/n): ").strip().lower()
        if confirmation == 'y':
            mycursor.execute("USE reg_finance")
            mycursor.execute(f"DELETE FROM {table_name}")
            db.commit()
            print(f"Все данные из таблицы '{table_name}' успешно удалены.")
        else:
            print("Удаление отменено.")
    except Error as e:
        print(f"Ошибка при удалении данных: {e}")
        db.rollback()
    finally:
        if db:
            db.close()

def add_data():
    mycursor, db = db_init.connect_to_data()
    if not mycursor:
        return

    try:
        table_name = input("Введите имя таблицы, в которую хотите добавить данные: ").strip()
        if table_name not in db_init.table_mapping:
            print(f"Таблица '{table_name}' не найдена.")
            return
        df = db_init.create_df(table_name)
        column_names = list(df.columns)
        values = []
        print("Введите значения для каждого столбца. Оставьте поле пустым для значения NULL.")
        for col in column_names:
            value = input(f"Введите значение для '{col}': ").strip()
            values.append(None if value == "" else db_init.numeric_convert(value))
        placeholders = ', '.join(['%s'] * len(column_names))
        sql = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({placeholders})"
        mycursor.execute("USE reg_finance")
        mycursor.execute(sql, values)
        db.commit()
        print("Данные успешно добавлены.")
    except Error as e:
        print(f"Ошибка при добавлении данных: {e}")
        db.rollback()
    finally:
        if db:
            db.close()

def edit_data():
    mycursor, db = db_init.connect_to_data()
    if not mycursor:
        return

    try:
        table_name = input("Введите имя таблицы, в которой хотите изменить данные: ").strip()
        if table_name not in db_init.table_mapping:
            print(f"Таблица '{table_name}' не найдена.")
            return

        df = db_init.create_df(table_name)
        if df is None or df.empty:
            print("Не удалось загрузить таблицу или таблица пуста.")
            return

        print("Доступные столбцы для редактирования:")
        for i, col in enumerate(df.columns):
            print(f"{i + 1}. {col}")

        column_index = int(input("Введите номер столбца для редактирования: ")) - 1
        column_name = df.columns[column_index]

        row_index = int(input("Введите код_региона или номер строки для редактирования: "))
        where_condition = f"код_региона = {row_index}" if df.index.name == 'код_региона' else f"1=1 LIMIT 1 OFFSET {row_index}"

        new_value = input(f"Введите новое значение для столбца '{column_name}': ").strip()
        new_value = None if new_value == "" else db_init.numeric_convert(new_value)

        sql = f"UPDATE {table_name} SET `{column_name}` = %s WHERE {where_condition}"
        mycursor.execute("USE reg_finance")
        mycursor.execute(sql, (new_value,))
        db.commit()
        print("Данные успешно обновлены.")
    except Error as e:
        print(f"Ошибка при обновлении данных: {e}")
        db.rollback()
    finally:
        if db:
            db.close()

def view_data():
    table_name = input("Введите имя таблицы, которую хотите просмотреть: ").strip()
    if table_name not in db_init.table_mapping:
        print(f"Таблица '{table_name}' не найдена.")
        return
    df = db_init.create_df(table_name)
    if df is None or df.empty:
        print("Не удалось загрузить таблицу или таблица пуста.")
        return
    print(df)

def show_region_data(region_code):
    mycursor, db = db_init.connect_to_data()
    if not mycursor:
        return

    try:
        mycursor.execute("USE reg_finance")
        tables = list(db_init.table_mapping.keys())
        for table in tables:
            mycursor.execute(f"SELECT * FROM {table} WHERE код_региона = %s", (region_code,))
            rows = mycursor.fetchall()
            if rows:
                print(f"Таблица: {table}")
                for row in rows:
                    print(row)
            else:
                print(f"Нет данных для региона {region_code} в таблице {table}")
    except Error as e:
        print(f"Ошибка при получении данных: {e}")
    finally:
        if db:
            db.close()

def show_multiple_regions_data(table_name, region_codes):
    mycursor, db = db_init.connect_to_data()
    if not mycursor:
        return

    try:
        mycursor.execute("USE reg_finance")
        placeholders = ', '.join(['%s'] * len(region_codes))
        mycursor.execute(f"SELECT * FROM {table_name} WHERE код_региона IN ({placeholders})", tuple(region_codes))
        rows = mycursor.fetchall()
        if rows:
            print(f"Таблица: {table_name}")
            for row in rows:
                print(row)
        else:
            print(f"Нет данных для указанных регионов в таблице {table_name}")
    except Error as e:
        print(f"Ошибка при получении данных: {e}")
    finally:
        if db:
            db.close()