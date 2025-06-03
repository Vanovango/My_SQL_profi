import db_init
import db_operations
import data_analysis

def main():
    db_init.execute_sql_script()

    while True:
        print("\n1. Редактирование и работа с БД")
        print("2. Анализ данных")
        print("0. Выход")
        choice = input("Выберите опцию: ").strip()

        if choice == "1":
            db_menu()
        elif choice == "2":
            analysis_menu()
        elif choice == "0":
            print("Выход из программы.")
            break
        else:
            print("Некорректный выбор. Попробуйте снова.")

def db_menu():
    while True:
        print("\na) Редактирование")
        print("b) Работа с БД")
        print("0. Назад")
        sub_choice = input("Выберите опцию: ").strip()

        if sub_choice == "a":
            edit_menu()
        elif sub_choice == "b":
            work_menu()
        elif sub_choice == "0":
            break
        else:
            print("Некорректный выбор. Попробуйте снова.")

def edit_menu():
    print("\ni. Удаление")
    print("ii. Добавление")
    print("iii. Изменение")
    edit_choice = input("Выберите опцию: ").strip()

    if edit_choice == "i":
        db_operations.delete_data()
    elif edit_choice == "ii":
        db_operations.add_data()
    elif edit_choice == "iii":
        db_operations.edit_data()
    else:
        print("Некорректный выбор.")

def work_menu():
    print("\ni. Показать 1 таблицу целиком")
    print("ii. Показать данные 1 региона во всех таблицах")
    print("iii. Показать данные нескольких регионов в 1 таблице")
    work_choice = input("Выберите опцию: ").strip()

    if work_choice == "i":
        db_operations.view_data()
    elif work_choice == "ii":
        region_code = input("Введите код региона: ").strip()
        db_operations.show_region_data(region_code)
    elif work_choice == "iii":
        table_name = input("Введите имя таблицы: ").strip()
        region_codes = input("Введите коды регионов через пробел: ").strip().split()
        db_operations.show_multiple_regions_data(table_name, region_codes)
    else:
        print("Некорректный выбор.")

def analysis_menu():
    print("\na) Анализ ключевых показателей для всех регионов")
    print("b) Корреляционный анализ")
    print("c) Анализ распределения")
    print("d) Фиксирование аномалий")
    print("e) Прогнозирование")
    analysis_choice = input("Выберите опцию: ").strip()

    if analysis_choice == "a":
        data_analysis.basic_stats()
    elif analysis_choice == "b":
        data_analysis.correlation_analysis()
    elif analysis_choice == "c":
        data_analysis.data_distribution_analysis()
    elif analysis_choice == "d":
        data_analysis.anomaly_detection()
    elif analysis_choice == "e":
        data_analysis.data_forecasting()
    else:
        print("Некорректный выбор.")

if __name__ == "__main__":
    main()