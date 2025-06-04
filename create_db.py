import mysql.connector
from mysql.connector import Error

def load_sql_dump(db_config, sql_file_path):
    """
    Открывает SQL-файл дампа и выполняет его команды для заполнения базы данных.

    :param db_config: Словарь с параметрами подключения к MySQL.
                      Пример: {
                          'host': '127.0.0.1',
                          'user': 'root',
                          'passwd': 'ваш_пароль',
                          'database': 'reg_finance'
                      }
    :param sql_file_path: Путь к SQL-файлу дампа (например, 'Dump20250603.sql').
    """
    connection = None
    cursor = None
    try:
        # Подключение к MySQL без указания базы данных для создания reg_finance
        temp_config = db_config.copy()
        temp_config.pop('database', None)  # Удаляем database из конфигурации
        connection = mysql.connector.connect(
            **temp_config,
            charset='utf8mb4',
            collation='utf8mb4_0900_ai_ci'
        )
        cursor = connection.cursor()

        # Проверка и создание базы данных reg_finance
        database_name = db_config.get('database', 'reg_finance')
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
        cursor.execute(f"USE `{database_name}`")
        connection.commit()
        print(f"База данных {database_name} выбрана или создана")

        # Чтение SQL-файла
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            sql_commands = file.read()

        # Разделение команд по разделителю ';'
        commands = sql_commands.split(';')

        # Выполнение каждой команды
        for command in commands:
            command = command.strip()
            if command:  # Пропуск пустых команд
                try:
                    cursor.execute(command)
                    connection.commit()
                    print(f"Команда выполнена: {command[:50]}...")
                except Error as e:
                    print(f"Ошибка при выполнении команды: {e}")
                    print(f"Команда: {command[:100]}...")
                    connection.rollback()

        print("Все команды из SQL-файла выполнены успешно")

    except Error as e:
        print(f"Произошла ошибка: {e}")
        if connection and connection.is_connected():
            connection.rollback()

    except FileNotFoundError:
        print(f"Файл не найден: {sql_file_path}")

    finally:
        # Закрытие соединения и курсора
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
            print("Соединение с MySQL закрыто")

# Пример использования
if __name__ == "__main__":
    # Конфигурация подключения
    db_config = {
        'host': '127.0.0.1',
        'user': 'root',
        'passwd': '2431',  # Замените на ваш пароль
        'database': 'reg_finance'
    }

    # Путь к SQL-файлу
    sql_file_path = 'Dump20250603.sql'

    # Вызов функции для загрузки дампа
    load_sql_dump(db_config, sql_file_path)