import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Model:
    def __init__(self, ts_df, group, periods, test=False):
        self.ts_df = ts_df
        self.group = group
        self.periods = periods
        self.test = test

    def filter_df(self):
        if self.test:
            df_filtered = (self.ts_df[self.ts_df['price_group'] == self.group]
                  .copy()
                  .sort_values('year_month'))
            df_filtered['date'] = df_filtered['year_month'].dt.to_timestamp()
            # Calculate cutoff date: last date - 24 months
            cutoff_date = df_filtered['date'].max() - pd.DateOffset(months=self.periods)
            # Split into two
            df_filtered_train = df_filtered[df_filtered['date'] <= cutoff_date]
            df_filtered_test = df_filtered[df_filtered['date'] > cutoff_date]
            df_filtered_train.drop(columns='date', inplace=True)
            df_filtered_test.drop(columns='date', inplace=True)
            return df_filtered_train, df_filtered_test
        else:
            df_filtered_train = (self.ts_df[self.ts_df['price_group'] == self.group]
                  .copy()
                  .sort_values('year_month'))
            df_filtered_test = pd.DataFrame()
            return df_filtered_train, df_filtered_test

    @staticmethod
    def normalize_index(df_filtered):
        if isinstance(df_filtered['year_month'].dtype, pd.PeriodDtype):
            return df_filtered['year_month'].dt.to_timestamp(how='start')
        else:
            return pd.to_datetime(df_filtered['year_month']).dt.to_period('M').dt.to_timestamp(how='start')

    @staticmethod
    def fill_nan(df_filtered, index_ts):
        y = df_filtered['avg_price'].astype(float).values
        s = pd.Series(y, index=index_ts).asfreq('MS')
        s = s.interpolate(limit_direction='both')
        return s.values, s.index

    @staticmethod
    def train_model(y_train, y_index):
        # Prophet requiere columnas ds (datetime) e y (valor)
        train = pd.DataFrame({'ds': y_index, 'y': y_train})

        # Modelo Prophet (mensual -> sin weekly/daily)
        model = Prophet(
            yearly_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.01,
            changepoint_range=0.8
        )
        model.fit(train)
        return train, model

    def prediction(self, model):
        future = model.make_future_dataframe(periods=self.periods, freq='MS')
        return model.predict(future)

    @staticmethod
    def validation_metrics(forecast, y_test, y_index):
        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100
        r2 = r2_score(y_test, forecast)

        # Print metrics
        print(f"MAE:  {mae:.2f}")
        print(f"MSE:  {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²:   {r2:.2f}")

        # Plot actual vs predicted
        plt.figure(figsize=(10, 5))
        plt.plot(y_index, y_test, label='Actual', marker='o')
        plt.plot(y_index, forecast, label='Predicted', marker='x')
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Forecast vs Actual")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot(self, train, forecast):
        plt.figure(figsize=(14, 7))
        plt.plot(train['ds'], train['y'], label='Histórico')
        future_tail = forecast[forecast['ds'] > train['ds'].max()]
        plt.plot(future_tail['ds'], future_tail['yhat'], label='Predicción')
        plt.fill_between(future_tail['ds'], future_tail['yhat_lower'], future_tail['yhat_upper'], alpha=0.25)
        plt.title(f'Prophet — Precio promedio mensual ({self.group})')
        plt.xlabel('Fecha')
        plt.ylabel('Precio promedio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def execute(self):
        df_filtered_train, df_filtered_test = self.filter_df()
        index_ts = self.normalize_index(df_filtered_train)
        y_train, y_index = self.fill_nan(df_filtered_train, index_ts)
        train, model = self.train_model(y_train, y_index)
        forecast = self.prediction(model)
        if self.test:
            index_ts_test = self.normalize_index(df_filtered_test)
            y_test, y_index = self.fill_nan(df_filtered_test, index_ts_test)
            self.validation_metrics(forecast[forecast['ds'] >= y_index.min()]['yhat'].values, y_test, y_index)
            pass
        else:
            self.plot(train, forecast)

