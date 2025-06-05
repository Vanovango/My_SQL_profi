import os

import db_init
import db_operations
import data_analysis

def main():
    print("💼 Добро пожаловать в систему анализа региональных финансов 📊")
    while True:
        print("\n🔧 Главное меню:")
        print("  1️⃣  Редактирование и работа с БД 🛠️")
        print("  2️⃣  Анализ данных 📈")
        print("  0️⃣  Выход ❌")
        choice = input("👉 Ваш выбор: ").strip()

        if choice == "1":
            db_menu()
        elif choice == "2":
            analysis_menu()
        elif choice == "0":
            print("👋 До свидания! Работа завершена.")
            break
        else:
            print("⚠️ Некорректный выбор. Попробуйте снова.")

def db_menu():
    while True:
        print("\n📁 Подменю базы данных:")
        print("  1️⃣  Редактирование ✏️")
        print("  2️⃣  Работа с БД 🗂️")
        print("  0️⃣  Назад 🔙")
        sub_choice = input("👉 Ваш выбор: ").strip()

        if sub_choice == "1":
            region_code = input("🔍 Введите код региона, с которым хотите работать: ").strip()
            if not region_code.isdigit():
                print("⚠️ Некорректный код региона.")
                continue
            edit_menu(region_code)
        elif sub_choice == "2":
            work_menu()
        elif sub_choice == "0":
            break
        else:
            print("⚠️ Некорректный выбор. Попробуйте снова.")

def edit_menu(region_code):
    while True:
        print(f"\n📝 Редактирование данных для региона 🔢 {region_code}:")
        print("  1️⃣  Удаление ❌")
        print("  2️⃣  Добавление ➕")
        print("  3️⃣  Изменение 🛠️")
        print("  0️⃣  Назад 🔙")
        edit_choice = input("👉 Ваш выбор: ").strip()

        if edit_choice == "1":
            db_operations.delete_data(region_code)
        elif edit_choice == "2":
            db_operations.add_data(region_code)
        elif edit_choice == "3":
            db_operations.edit_data(region_code)
        elif edit_choice == "0":
            return
        else:
            print("⚠️ Некорректный выбор.")

def work_menu():
    print("\n📊 Работа с данными:")
    print("  1️⃣  Показать таблицу полностью 🧾")
    print("  2️⃣  Данные региона по всем таблицам 🌍")
    print("  3️⃣  Данные по регионам из 1 таблицы 📋")
    print("  0️⃣  Назад 🔙")

    work_choice = input("👉 Ваш выбор: ").strip()

    if work_choice == "1":
        db_operations.view_data()
    elif work_choice == "2":
        region_code = input("🔍 Введите код региона: ").strip()
        db_operations.show_region_data(region_code)
    elif work_choice == "3":
        table_name = input("📁 Введите имя таблицы: ").strip()
        region_codes = input("🔢 Введите коды регионов через пробел: ").strip().split()
        db_operations.show_multiple_regions_data(table_name, region_codes)
    elif work_choice == "0":
        return
    else:
        print("⚠️ Некорректный выбор.")

def analysis_menu():
    print("\n📈 Меню анализа данных:")
    print("  1️⃣  Ключевые показатели 📌")
    print("  2️⃣  Корреляция 🔗")
    print("  3️⃣  Распределение 📊")
    print("  4️⃣  Аномалии 🚨")
    print("  5️⃣  Прогнозирование 🔮")
    print("  0️⃣  Назад 🔙")

    analysis_choice = input("👉 Ваш выбор: ").strip()

    if analysis_choice == "1":
        data_analysis.basic_stats()
    elif analysis_choice == "2":
        data_analysis.correlation_analysis()
    elif analysis_choice == "3":
        data_analysis.data_distribution_analysis()
    elif analysis_choice == "4":
        data_analysis.anomaly_detection()
    elif analysis_choice == "5":
        data_analysis.data_forecasting()
    elif analysis_choice == "0":
        return
    else:
        print("⚠️ Некорректный выбор.")

if __name__ == "__main__":
    main()
