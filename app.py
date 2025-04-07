import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score


if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None

class PriceForecaster:
    def forecast(self, period_percentages, min_price, max_price):
        
        base_trend = np.linspace(min_price, max_price, len(period_percentages))
        noise = np.random.uniform(-0.05, 0.05, len(period_percentages)) * (max_price - min_price)
        return base_trend + noise

class MonteCarloSimulator:
    def __init__(self, sales_forecaster):
        self.sales_forecaster = sales_forecaster
        self.price_forecaster = PriceForecaster()
    
    def run_simulation(self, project_type, total_area, duration, min_price, max_price, 
                      cost_official, unfin_cost, num_simulations=10000, 
                      start_month=1, months_completed=None, actual_sales=None, period_pct=None):
        
        
        if project_type == 'new':
            forecast_df = self.sales_forecaster.forecast_new_project(
                total_area, duration, start_month)
            sales_forecast = forecast_df['area'].values
            period_pct = forecast_df['period_%'].values
        elif project_type == 'ongoing':
            if months_completed is None or actual_sales is None or period_pct is None:
                raise ValueError("Для текущих проектов укажите months_completed, actual_sales и period_pct")
            forecast_df, _, _ = self.sales_forecaster.forecast_ongoing_project(
                total_area, duration, start_month, months_completed, actual_sales, period_pct)
            sales_forecast = forecast_df['area'].values
            period_pct = forecast_df['period_%'].values
        
        
        price_forecast = self.price_forecaster.forecast(period_pct, min_price, max_price)
        
        
        sales_lower = sales_forecast * 0.9
        sales_upper = sales_forecast * 1.1
        price_lower = price_forecast * 0.9
        price_upper = price_forecast * 1.1
        
        
        Capex_mean = (cost_official - unfin_cost) / duration
        Capex_lower = Capex_mean * 0.9
        Capex_upper = Capex_mean * 1.1
        
        
        total_cash_flows = []
        for _ in range(num_simulations):
            
            sales_coef = np.random.uniform(0.9, 1.1)
            price_coef = np.random.uniform(0.9, 1.1)
            capex_coef = np.random.uniform(0.9, 1.1)
            
           
            sales_sim = sales_forecast * sales_coef
            price_sim = price_forecast * price_coef
            capex_sim = Capex_mean * capex_coef
            
            CFi = sales_sim * price_sim - capex_sim
            total_cash_flows.append(np.sum(CFi))

        total_cash_flows = np.array(total_cash_flows)
        
        
        failure_rate = np.sum(total_cash_flows <= 0) / num_simulations
        mean_cash_flow = np.mean(total_cash_flows)
        std_dev_cash_flow = np.std(total_cash_flows)
        
        
        self._plot_results(sales_forecast, sales_lower, sales_upper,
                         price_forecast, price_lower, price_upper,
                         total_cash_flows, mean_cash_flow,
                         std_dev_cash_flow, failure_rate,
                         project_type, duration)
        
        return {
            'failure_rate': failure_rate,
            'mean_cash_flow': mean_cash_flow,
            'std_dev_cash_flow': std_dev_cash_flow,
            'min_cash_flow': np.min(total_cash_flows),
            'max_cash_flow': np.max(total_cash_flows)
        }
    
    def _plot_results(self, sales_forecast, sales_lower, sales_upper,
                     price_forecast, price_lower, price_upper,
                     total_cash_flows, mean_cf,
                     std_dev_cf, failure_rate,
                     project_type, duration):
        
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(range(1, duration+1), sales_forecast, 
                'o-', color='blue', label='Прогнозируемые продажи', markersize=6)
        ax1.fill_between(range(1, duration+1), 
                        sales_lower, 
                        sales_upper,
                        color='blue', alpha=0.2, 
                        label='±10% диапазон продаж')
        ax1.set_title(f"{project_type.capitalize()} Проект - Прогноз продаж с постоянным 10% диапазоном")
        ax1.set_xlabel("Месяц")
        ax1.set_ylabel("Проданная площадь (м²)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        st.pyplot(fig1)
        
       
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(range(1, duration+1), price_forecast,
                'o-', color='red', label='Прогнозируемые цены', markersize=6)
        ax2.fill_between(range(1, duration+1),
                        price_lower,
                        price_upper,
                        color='red', alpha=0.2,
                        label='±10% диапазон цен')
        ax2.set_title(f"{project_type.capitalize()} Проект - Прогноз цен с постоянным 10% диапазоном")
        ax2.set_xlabel("Месяц")
        ax2.set_ylabel("Цена за м²")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        st.pyplot(fig2)
        
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        counts, bins, _ = ax3.hist(total_cash_flows, bins=50, 
                                  density=True, alpha=0.6,
                                  color='green', edgecolor='black')
        
        xmin, xmax = ax3.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_cf, std_dev_cf)
        ax3.plot(x, p, 'k', linewidth=2, label='Нормальное распределение')
        
        ax3.axvline(mean_cf, color='red', linestyle='--', 
                   label=f'Среднее ({mean_cf:,.0f})')
        ax3.axvline(0, color='black', linestyle='-',
                   label='Точка безубыточности')
        
        failure_x = np.linspace(xmin, 0, 50)
        failure_p = stats.norm.pdf(failure_x, mean_cf, std_dev_cf)
        ax3.fill_between(failure_x, failure_p, color='red', alpha=0.3,
                        label=f'Риск неудачи ({failure_rate:.1%})')
        
        ax3.set_title(f"Результаты моделирования Монте-Карло\n{project_type.capitalize()} Проект - Распределение денежных потоков")
        ax3.set_xlabel("Общий денежный поток")
        ax3.set_ylabel("Плотность вероятности")
        ax3.legend()
        ax3.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig3)

class ProjectSalesForecaster:
    def __init__(self, data_path="data_z.xlsx"):
        self.df = pd.read_excel(data_path)
        self.model = None
        self.features = None
        self.train_mae = None
        self.test_mae = None
        self.train_r2 = None
        self.test_r2 = None
        self.test_results = None
        self.monte_carlo = None
        
    def preprocess_data(self, df):
        required_cols = ["project_name", "period_%", "area", "record_date"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Отсутствуют необходимые столбцы")
        
        df = df[required_cols].copy()
        df["record_date"] = pd.to_datetime(df["record_date"])
        
        project_stats = df.groupby("project_name").agg(
            total_area=("area", "sum"),
            duration=("record_date", "count"),
            start_date=("record_date", "min"),
            end_date=("record_date", "max")
        ).reset_index()
        
        df = df.merge(project_stats, on="project_name")
        
        df["time_decay"] = 1 / (1 + df["period_%"])
        df["portion_sold"] = df["area"] / df["total_area"]
        df["month"] = df["record_date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"]/12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"]/12)
        df["project_month"] = df.groupby("project_name").cumcount() + 1
        
        return df
    
    def split_data(self, df):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(df, groups=df["project_name"]))
        return df.iloc[train_idx], df.iloc[test_idx]
    
    def train_model(self):
        processed_data = self.preprocess_data(self.df)
        train_data, test_data = self.split_data(processed_data)
        
        self.features = ["period_%", "total_area", "duration", "time_decay", 
                        "month_sin", "month_cos", "project_month"]
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ])
        
        model.fit(train_data[self.features], train_data["portion_sold"])
        self.model = model
        
      
        train_pred = model.predict(train_data[self.features]) * train_data["total_area"]
        self.train_mae = mean_absolute_error(train_data["area"], train_pred)
        self.train_r2 = r2_score(train_data["area"], train_pred)
        
       
        test_pred = model.predict(test_data[self.features]) * test_data["total_area"]
        self.test_mae = mean_absolute_error(test_data["area"], test_pred)
        self.test_r2 = r2_score(test_data["area"], test_pred)
        
       
        self.test_results = self._evaluate_test_projects(test_data)
        
        
        self.monte_carlo = MonteCarloSimulator(self)
        
        return self
    
    def _evaluate_test_projects(self, test_data, num_projects=5):
        test_projects = test_data['project_name'].unique()[:num_projects]
        results = []
        
        fig, axes = plt.subplots(num_projects, 1, figsize=(15, 3*num_projects))
        if num_projects == 1:
            axes = [axes]
        
        for i, (project, ax) in enumerate(zip(test_projects, axes), 1):
            project_data = test_data[test_data["project_name"] == project]
            total_area = project_data["total_area"].iloc[0]
            
            pred = self.model.predict(project_data[self.features]) * total_area
            mae = mean_absolute_error(project_data["area"], pred)
            r2 = r2_score(project_data["area"], pred)
            
            results.append({
                'project_name': project,
                'mae': mae,
                'r2_score': r2,
                'avg_area': project_data["area"].mean(),
                'data_points': len(project_data),
                'total_area': total_area
            })
            
            ax.plot(project_data["period_%"], project_data["area"], 
                   'o-', color='#1f77b4', label='Фактические продажи', markersize=5, linewidth=1.5)
            ax.plot(project_data["period_%"], pred, 
                   's--', color='#ff7f0e', label='Прогнозируемые', markersize=4, linewidth=1.5)
            
            ax.set_title(f"{project}\nMAE: {mae:.1f} м² | R²: {r2:.2f} | Общая площадь: {total_area:.0f} м²", 
                        fontsize=10, pad=10)
            ax.set_xlabel("Завершение проекта (period_%)", fontsize=8)
            ax.set_ylabel("Проданная площадь (м²)", fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.7)
            
            if i == 1:
                ax.legend(fontsize=8, framealpha=0.9)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)
        
        return pd.DataFrame(results)
    
    def print_model_metrics(self):
        st.subheader("Метрики производительности модели")
        st.write(f"**MAE на обучающей выборке:** {self.train_mae:.2f} м²")
        st.write(f"**MAE на тестовой выборке:** {self.test_mae:.2f} м²")
        st.write(f"**R² на обучающей выборке:** {self.train_r2:.3f}")
        st.write(f"**R² на тестовой выборке:** {self.test_r2:.3f}")
        st.write(f"**Соотношение MAE обучающей/тестовой:** {self.train_mae/self.test_mae:.2f}")
        
        if self.test_results is not None:
            st.subheader("Производительность на тестовых проектах")
            st.dataframe(self.test_results.sort_values('mae').round(2))
            
            st.subheader("Статистика тестовой выборки")
            st.write(f"Средний MAE по тестовым проектам: {self.test_results['mae'].mean():.2f} м²")
            st.write(f"Медианный MAE по тестовым проектам: {self.test_results['mae'].median():.2f} м²")
            st.write(f"Лучший показатель R²: {self.test_results['r2_score'].max():.2f}")
            st.write(f"Худший показатель R²: {self.test_results['r2_score'].min():.2f}")
    
    def forecast_new_project(self, total_area, duration, start_month=1):
        months = np.arange(1, duration+1)
        period_pct = months/duration
        
        project_data = pd.DataFrame({
            "period_%": period_pct,
            "total_area": total_area,
            "duration": duration,
            "time_decay": 1 / (1 + period_pct),
            "month": (start_month - 1 + months) % 12 + 1,
            "month_sin": np.sin(2 * np.pi * (start_month - 1 + months)/12),
            "month_cos": np.cos(2 * np.pi * (start_month - 1 + months)/12),
            "project_month": months
        })
        
        project_data["area"] = self.model.predict(project_data[self.features]) * total_area
        
        self._plot_project_forecast(project_data, project_name=f"Прогноз нового проекта (Площадь: {total_area}м², Длительность: {duration} месяцев)")
        
        return project_data
    
    def forecast_ongoing_project(self, total_area, duration, start_month, months_completed, actual_sales, period_pct):
       
        
      
        if len(actual_sales) != months_completed:
            raise ValueError(f"Длина actual_sales ({len(actual_sales)}) должна соответствовать months_completed ({months_completed})")
        if len(period_pct) != months_completed:
            raise ValueError(f"Длина period_pct ({len(period_pct)}) должна соответствовать months_completed ({months_completed})")
        
        months = np.arange(1, months_completed+1)
        
        project_data = pd.DataFrame({
            "period_%": period_pct,
            "area": actual_sales,
            "total_area": total_area,
            "duration": duration,
            "time_decay": 1 / (1 + np.array(period_pct)),
            "month": (start_month - 1 + months) % 12 + 1,
            "month_sin": np.sin(2 * np.pi * (start_month - 1 + months)/12),
            "month_cos": np.cos(2 * np.pi * (start_month - 1 + months)/12),
            "project_month": months
        })
        
       
        project_data["predicted"] = self.model.predict(project_data[self.features]) * total_area
        mae = mean_absolute_error(project_data["area"], project_data["predicted"])
        r2 = r2_score(project_data["area"], project_data["predicted"])
        
        
        if months_completed < duration:
            remaining_months = duration - months_completed
            future_months = np.arange(months_completed+1, duration+1)
            future_period_pct = np.linspace(period_pct[-1] + (1-period_pct[-1])/remaining_months, 1, remaining_months)
            
            future_data = pd.DataFrame({
                "period_%": future_period_pct,
                "total_area": total_area,
                "duration": duration,
                "time_decay": 1 / (1 + future_period_pct),
                "month": (start_month - 1 + future_months) % 12 + 1,
                "month_sin": np.sin(2 * np.pi * (start_month - 1 + future_months)/12),
                "month_cos": np.cos(2 * np.pi * (start_month - 1 + future_months)/12),
                "project_month": future_months
            })
            
            future_data["area"] = self.model.predict(future_data[self.features]) * total_area
            future_data["predicted"] = future_data["area"]
            
            full_data = pd.concat([
                project_data.assign(type="actual"),
                future_data.assign(type="forecast")
            ])
        else:
            full_data = project_data.assign(type="actual")
        
        self._plot_ongoing_project_forecast(full_data, months_completed, 
                                         f"Текущий проект (Завершено: {months_completed}/{duration} месяцев)",
                                         mae)
        
        return full_data, mae, r2

    def _plot_project_forecast(self, project_data, project_name):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(project_data["period_%"], project_data["area"], 
                'o-', color='blue', label='Прогнозируемые продажи', markersize=8)
        
        ax.set_title(f"{project_name}\nОбщая площадь: {project_data['total_area'].iloc[0]:.0f}м² | Длительность: {project_data['duration'].iloc[0]} месяцев", 
                 fontsize=14)
        ax.set_xlabel("Завершение проекта (period_%)", fontsize=12)
        ax.set_ylabel("Проданная площадь (м²)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        st.pyplot(fig)
    
    def _plot_ongoing_project_forecast(self, forecast_df, num_actual, project_name, mae):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        actual_data = forecast_df[forecast_df["type"] == "actual"]
        ax.plot(actual_data["period_%"], actual_data["area"], 
                'o-', color='green', label='Фактические продажи', markersize=8)
        
        ax.plot(actual_data["period_%"], actual_data["predicted"], 
                'x--', color='red', label='Прогнозируемые продажи', markersize=8)
        
        if len(forecast_df) > num_actual:
            forecast_data = forecast_df[forecast_df["type"] == "forecast"]
            
            ax.plot([actual_data["period_%"].iloc[-1], forecast_data["period_%"].iloc[0]], 
                    [actual_data["area"].iloc[-1], forecast_data["area"].iloc[0]], 
                    '--', color='blue', alpha=0.5)
            
            ax.plot(forecast_data["period_%"], forecast_data["area"], 
                    'o--', color='blue', label='Прогноз продаж', markersize=8)
        
        ax.set_title(f"{project_name}\nMAE: {mae:.2f} м²", fontsize=14)
        ax.set_xlabel("Завершение проекта (period_%)", fontsize=12)
        ax.set_ylabel("Проданная площадь (м²)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        st.pyplot(fig)

def main():
    st.title("🏠 Прогнозирование продаж жилых проектов и анализ рисков")
    st.markdown("""
    Этот инструмент помогает прогнозировать продажи и анализировать финансовые риски жилых проектов с использованием:
    - **Машинного обучения** (XGBoost) для прогнозирования продаж
    - **Моделирования Монте-Карло** для оценки рисков
    """)
    
    
    if st.session_state.forecaster is None:
        with st.spinner('Загрузка и обучение модели...'):
            st.session_state.forecaster = ProjectSalesForecaster().train_model()
    
   
    option = st.sidebar.radio("Выберите опцию", ["Новый проект", "Текущий проект", "Производительность модели"])
    
    if option == "Новый проект":
        st.header("Прогноз для нового проекта")
        
        with st.form("new_project_form"):
            col1, col2 = st.columns(2)
            total_area = col1.number_input("Общая площадь (м²)", min_value=1000, value=5000)
            duration = col2.number_input("Длительность (месяцев)", min_value=1, value=12)
            
            col1, col2 = st.columns(2)
            min_price = col1.number_input("Минимальная цена за м²", min_value=100000, value=350000)
            max_price = col2.number_input("Максимальная цена за м²", min_value=100000, value=400000)
            
            col1, col2 = st.columns(2)
            cost_official = col1.number_input("Общая официальная стоимость", min_value=0, value=3000000000)
            unfin_cost = col2.number_input("Незавершенные затраты", min_value=0, value=50000000)
            
            start_month = st.selectbox("Начальный месяц", range(1, 13), index=0)
            
            submitted = st.form_submit_button("Запустить прогноз и моделирование")
        
        if submitted:
            with st.spinner('Выполняется анализ...'):
               
                st.subheader("Прогноз продаж")
                forecast_df = st.session_state.forecaster.forecast_new_project(
                    total_area, duration, start_month)
                
                
                st.subheader("Результаты моделирования Монте-Карло")
                results = st.session_state.forecaster.monte_carlo.run_simulation(
                    project_type='new',
                    total_area=total_area,
                    duration=duration,
                    min_price=min_price,
                    max_price=max_price,
                    cost_official=cost_official,
                    unfin_cost=unfin_cost,
                    start_month=start_month
                )
                
                
                st.success("Анализ завершен!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Вероятность неудачи", f"{results['failure_rate']:.1%}")
                col2.metric("Ожидаемый денежный поток", f"{results['mean_cash_flow']:,.0f} тг")
                col3.metric("Риск (Станд. отклонение)", f"{results['std_dev_cash_flow']:,.0f} тг")
                
                st.write(f"**Лучший сценарий:** {results['max_cash_flow']:,.0f} тг")
                st.write(f"**Худший сценарий:** {results['min_cash_flow']:,.0f} тг")
    
    elif option == "Текущий проект":
        st.header("Прогноз для текущего проекта")
        
        with st.form("ongoing_project_form"):
            col1, col2 = st.columns(2)
            total_area = col1.number_input("Общая площадь (м²)", min_value=1000, value=5366)
            duration = col2.number_input("Длительность (месяцев)", min_value=1, value=11)
            
            col1, col2 = st.columns(2)
            min_price = col1.number_input("Минимальная цена за м²", min_value=100000, value=350000)
            max_price = col2.number_input("Максимальная цена за м²", min_value=100000, value=370000)
            
            col1, col2 = st.columns(2)
            cost_official = col1.number_input("Общая официальная стоимость", min_value=0, value=2228232541)
            unfin_cost = col2.number_input("Незавершенные затраты", min_value=0, value=33160000)
            
            months_completed = st.number_input("Завершенные месяцы", min_value=1, max_value=duration-1, value=5)
            
           
            st.subheader("Исторические данные")
            col1, col2 = st.columns(2)
            actual_sales = col1.text_input("Фактические продажи (через запятую)", value="400,420,450,480,510")
            period_pct = col2.text_input("Period % (через запятую)", value="0.1,0.2,0.3,0.4,0.5")
            
            start_month = st.selectbox("Начальный месяц", range(1, 13), index=0)
            
            submitted = st.form_submit_button("Запустить прогноз и моделирование")
        
        if submitted:
            with st.spinner('Выполняется анализ...'):
                
                try:
                    sales_list = [float(x.strip()) for x in actual_sales.split(',')]
                    period_list = [float(x.strip()) for x in period_pct.split(',')]
                    
                    if len(sales_list) != months_completed:
                        st.error(f"Пожалуйста, укажите ровно {months_completed} значений продаж!")
                        return
                    if len(period_list) != months_completed:
                        st.error(f"Пожалуйста, укажите ровно {months_completed} значений period_%!")
                        return
                except Exception as e:
                    st.error(f"Неверный формат ввода: {str(e)}")
                    return
                
                
                st.subheader("Прогноз продаж")
                forecast_df, mae, r2 = st.session_state.forecaster.forecast_ongoing_project(
                    total_area, duration, start_month, 
                    months_completed, sales_list, period_list)
                
               
                st.subheader("Результаты моделирования Монте-Карло")
                results = st.session_state.forecaster.monte_carlo.run_simulation(
                    project_type='ongoing',
                    total_area=total_area,
                    duration=duration,
                    min_price=min_price,
                    max_price=max_price,
                    cost_official=cost_official,
                    unfin_cost=unfin_cost,
                    start_month=start_month,
                    months_completed=months_completed,
                    actual_sales=sales_list,
                    period_pct=period_list
                )
                
               
                st.success("Анализ завершен!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Вероятность неудачи", f"{results['failure_rate']:.1%}")
                col2.metric("Ожидаемый денежный поток", f"{results['mean_cash_flow']:,.0f} тг")
                col3.metric("Риск (Станд. отклонение)", f"{results['std_dev_cash_flow']:,.0f} тг")
                
                st.write(f"**Лучший сценарий:** {results['max_cash_flow']:,.0f} тг")
                st.write(f"**Худший сценарий:** {results['min_cash_flow']:,.0f} тг")
                st.write(f"**Точность модели (MAE):** {mae:.2f} м² | **R² Score:** {r2:.2f}")
    
    elif option == "Производительность модели":
        st.header("Производительность модели")
        
        if st.session_state.forecaster is None:
            st.warning("Пожалуйста, сначала загрузите и обучите модель")
            return
        
        
        st.session_state.forecaster.print_model_metrics()
        
        
        st.subheader("Сравнение прогнозируемых и фактических значений")
        
        
        processed_data = st.session_state.forecaster.preprocess_data(st.session_state.forecaster.df)
        _, test_data = st.session_state.forecaster.split_data(processed_data)
        
        
        test_projects = test_data['project_name'].unique()[:3]  # Показать первые 3 проекта
        
        for project in test_projects:
            project_data = test_data[test_data["project_name"] == project]
            total_area = project_data["total_area"].iloc[0]
            
           
            pred = st.session_state.forecaster.model.predict(project_data[st.session_state.forecaster.features]) * total_area
            mae = mean_absolute_error(project_data["area"], pred)
            r2 = r2_score(project_data["area"], pred)
            
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(project_data["period_%"], project_data["area"], 
                   'o-', color='blue', label='Фактические продажи', markersize=5)
            ax.plot(project_data["period_%"], pred, 
                   's--', color='red', label='Прогнозируемые продажи', markersize=4)
            
            ax.set_title(f"Проект: {project}\nMAE: {mae:.1f} м² | R²: {r2:.2f}")
            ax.set_xlabel("Завершение проекта (period_%)")
            ax.set_ylabel("Проданная площадь (м²)")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
            st.pyplot(fig)

if __name__ == "__main__":
    main()