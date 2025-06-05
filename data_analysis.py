import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import db_init

def choose_table(selected_method):
    if selected_method == "5":
        print('Выберите две или более таблиц для анализа взаимосвязей.')
        selected_tables = input('Введите сокращения таблиц через пробел: ').strip().split()
        dfs = []
        for table in selected_tables:
            if table in db_init.table_mapping:
                dfs.append(db_init.table_mapping[table])
            else:
                print(f"Таблица '{table}' не найдена.")
        if len(dfs) < 2:
            print("Необходимо выбрать как минимум две таблицы для анализа взаимосвязей.")
            return None
        full_names = [db_init.table_fullname.get(tbl, "Неизвестное название") for tbl in selected_tables]
        print("\nВы выбрали следующие таблицы:")
        print(", ".join(full_names))
        return dfs
    else:
        print('Выберите одну таблицу для анализа.')
        selected_table = input('Введите сокращение таблицы: ').strip()
        if selected_table in db_init.table_mapping:
            df = db_init.table_mapping[selected_table]
            full_name = db_init.table_fullname.get(selected_table, "Неизвестное название")
            print(f"\nВы выбрали таблицу: {full_name}")
            return df
        else:
            print(f"Таблица '{selected_table}' не найдена.")
            return None

def basic_stats():
    print("Общая статистика данных")
    df = choose_table("1")
    if df is None or df.empty:
        print("Выбор таблицы завершился неудачей или таблица пуста.")
        return

    if 'код_региона' in df.columns:
        reg_name_df = db_init.create_df('коды_регионов')
        df['название_региона'] = df['код_региона'].apply(lambda reg: db_init.reg_name.loc[reg, 'название'] if reg in db_init.reg_name.index else "Регион не найден")
        df.drop(columns=['код_региона'], inplace=True)

    print("\nПервая часть данных (первые 5 строк):")
    print(df.head(), "\n")
    print("Базовая статистическая информация (числовые столбцы):")
    print(df.describe(include=[np.number]), "\n")

    numerical_columns = df.select_dtypes(include=[np.number]).columns
    if len(numerical_columns) > 0:
        stats = {
            "Среднее значение": df[numerical_columns].mean(),
            "Медиана": df[numerical_columns].median(),
            "Минимум": df[numerical_columns].min(),
            "Максимум": df[numerical_columns].max(),
            "Ст. отклонение": df[numerical_columns].std(),
        }
        stats_df = pd.DataFrame(stats)
        plt.figure(figsize=(10, 6))
        for stat_name in stats.keys():
            plt.plot(stats_df.index, stats_df[stat_name], marker='o', label=stat_name)
        plt.title("Сводная статистика для числовых столбцов", fontsize=16)
        plt.xlabel("Столбцы", fontsize=14)
        plt.ylabel("Значения", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Показатели", fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("В выбранной таблице отсутствуют числовые данные для построения графика.")

def correlation_analysis():
    print("Корреляционный анализ по таблицам со средними значениями за выбранный год.")
    selected_year = input("Введите год для анализа (например, 2021): ").strip()
    try:
        selected_year = int(selected_year)
    except ValueError:
        print("Ошибка: Введен некорректный год.")
        return

    year_col = str(selected_year)
    yearly_dataframes = {}
    for table_name, df in db_init.table_mapping.items():
        if df is None or year_col not in df.columns:
            continue
        col = df[year_col]
        if pd.api.types.is_numeric_dtype(col):
            yearly_dataframes[table_name] = df[[year_col]].copy()
            yearly_dataframes[table_name].rename(columns={year_col: table_name}, inplace=True)

    if not yearly_dataframes:
        print(f"Нет таблиц с данными за {selected_year} год.")
        return

    merged_df = list(yearly_dataframes.values())[0]
    for df in list(yearly_dataframes.values())[1:]:
        merged_df = merged_df.join(df, how='outer')

    correlation_matrix = merged_df.corr(method='pearson')
    print(f"\nКорреляционная матрица по таблицам за {selected_year} год:")
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title(f"Корреляция данных за {selected_year} год", fontsize=16)
    table_labels = [db_init.table_fullname.get(table, table) for table in correlation_matrix.columns]
    plt.yticks(ticks=np.arange(len(table_labels)), labels=table_labels, rotation=0, fontsize=8)
    plt.xticks(ticks=np.arange(len(table_labels)), labels=table_labels, rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.show()

def data_distribution_analysis():
    print("Анализ распределения данных")
    selected_year = input("Введите год для анализа (например, 2021): ").strip()
    try:
        selected_year = int(selected_year)
    except ValueError:
        print("Ошибка: Введен некорректный год.")
        return

    df = choose_table("3")
    if df is None or df.empty or str(selected_year) not in df.columns:
        print("Не удалось выбрать таблицу, таблица пуста или данных за год нет.")
        return

    plt.figure(figsize=(12, 6))
    sns.histplot(df[str(selected_year)].dropna(), kde=True)
    plt.title(f"Распределение '{selected_year}' (Гистограмма)")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.show()

    print(f"\nСтатистики для столбца '{selected_year}':")
    print(df[str(selected_year)].describe())

def anomaly_detection():
    print("Детекция аномалий")
    selected_year = input("Введите год для анализа (например, 2021): ").strip()
    try:
        selected_year = int(selected_year)
    except ValueError:
        print("Ошибка: Введен некорректный год.")
        return

    df = choose_table("4")
    if df is None or df.empty or str(selected_year) not in df.columns:
        print("Не удалось выбрать таблицу, таблица пуста или данных за год нет.")
        return

    z_scores = np.abs(stats.zscore(df[str(selected_year)].dropna()))
    threshold = 3
    anomalies = df[str(selected_year)].dropna()[z_scores > threshold]
    print(f"\nАномалии в столбце '{selected_year}' (Z-score > {threshold}):")
    print(anomalies)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[str(selected_year)])
    plt.title(f"Boxplot для '{selected_year}' (с аномалиями)")
    plt.xlabel("Значение")
    plt.show()

def data_forecasting():
    print("Прогнозирование данных")
    df = choose_table("6")
    if df is None or df.empty:
        print("Не удалось выбрать таблицу или таблица пуста.")
        return

    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    year_columns = [col for col in numerical_columns if col.isdigit()]
    if len(year_columns) < 2:
        print("Недостаточно данных по годам для прогнозирования.")
        return

    forecast_years = [int(year) for year in input("Введите годы для прогнозирования через пробел: ").strip().split()]
    changes = {index: {year_columns[i]: row[year_columns[i]] - row[year_columns[i-1]] for i in range(1, len(year_columns))}
               for index, row in df.iterrows() if not row[year_columns].isnull().any()}
    if not changes:
        print("Недостаточно данных для расчета изменений.")
        return

    changes_df = pd.DataFrame.from_dict(changes, orient='index').dropna(axis=1, how='all')
    target_column = year_columns[-1]
    feature_columns = [col for col in changes_df.columns if col != target_column]

    X = changes_df[feature_columns].fillna(changes_df[feature_columns].mean())
    y = changes_df[target_column].fillna(changes_df[target_column].mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nСреднеквадратичная ошибка (MSE): {mean_squared_error(y_test, y_pred):.2f}")

    last_year_data = df[year_columns[-1]].mean()
    current_value = last_year_data
    predicted_values = {}
    for forecast_year in forecast_years:
        forecast_data = pd.DataFrame([X.mean()], columns=X.columns)
        predicted_change = model.predict(forecast_data)[0]
        current_value += predicted_change
        predicted_values[forecast_year] = current_value
        print(f"Прогнозируемое значение для {forecast_year}: {current_value:.2f}")

    years = [int(col) for col in year_columns] + forecast_years
    data = df[year_columns].mean().tolist() + [predicted_values[year] for year in forecast_years]
    plt.figure(figsize=(12, 6))
    plt.plot(years, data, marker='o', linestyle='-', color='blue', label='Исторические данные')
    for forecast_year in forecast_years:
        plt.scatter(forecast_year, predicted_values[forecast_year], color='red', s=100, label='Прогноз' if forecast_year == forecast_years[0] else "")
    plt.title("Прогнозирование данных")
    plt.xlabel("Год")
    plt.ylabel("Значение")
    plt.xticks(years)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()