import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import datetime

class forecasting_ML():
    
    def __init__(self, model, y_test):
        self.model = model
        self.y_test = y_test
    
    def forecast_to_check_quality(self, X_test, plot=0, check=0):
        y_pred_test = []
        for i in range(len(X_test)):
    
            pred = self.model.predict(X_test.iloc[i])

            month = X_test['month'][i]
            propertyType = X_test['propertyType'][i]
            bedrooms = X_test['bedrooms'][i]
            postcode = X_test['postcode'][i]

            y_pred_test.append(pred)

            for j in range(len(X_test)):

                if X_test['propertyType'][j] == propertyType\
                and X_test['bedrooms'][j] == bedrooms\
                and X_test['postcode'][j] == postcode:\

                    if X_test['month'][j] == month + 1 and X_test['lag1'][j] == 0:
                        X_test['lag1'][j] = pred

                    elif X_test['month'][j] == month + 2 and X_test['lag2'][j] == 0:
                        X_test['lag2'][j] = pred

                    elif X_test['month'][j] == month + 3 and X_test['lag3'][j] == 0:
                        X_test['lag3'][j] = pred

                    elif X_test['month'][j] == month + 6 and X_test['lag6'][j] == 0:
                        X_test['lag6'][j] = pred
        if check == 1:
            
            print(f'MAPE: {np.round(mean_absolute_percentage_error(np.exp(self.y_test), np.exp(y_pred_test)) * 100, 2)}%')
            print()
            print(f'MAE: {mean_absolute_error(np.exp(self.y_test), np.exp(y_pred_test))}')
            
        if plot == 1:
            self.plot_preds(y_pred_test)
            
        return y_pred_test
    
    def plot_preds(self, y_pred_test):
            
        compare_preds_test_df = pd.DataFrame([self.y_test.tolist(), y_pred_test]).T
        compare_preds_test_df.columns = ['real', 'prediction']
        compare_preds_test_df = compare_preds_test_df.sample(frac=.15)
        compare_preds_test_df = compare_preds_test_df.reset_index()
        compare_preds_test_df = compare_preds_test_df.drop('index', axis=1)
        compare_preds_test_df.plot(legend=True)
        plt.xlabel('index')
        plt.ylabel('value')
        plt.title('Compare predictions and real values on the TEST set')
        plt.show()    

    
    def create_dataset_for_predictions(self, dataset, bed_counts, postcodes, property_types, period):
        last_date = dataset['date'].max()
        last_date = pd.to_datetime(last_date)
        last_date += pd.DateOffset(months=1)

        periods = pd.date_range(start=last_date, periods=period, freq='MS')

        combinations = pd.DataFrame(columns=['date', 'bedrooms', 'postcode', 'propertyType'])

        for date in periods:
            for bedrooms in bed_counts:
                for postcode in postcodes:
                    for property_type in property_types:
                        combinations = combinations.append({
                            'date': date,
                            'bedrooms': bedrooms,
                            'postcode': postcode,
                            'propertyType': property_type
                        }, ignore_index=True)
        combinations['lag1'] = 0
        combinations['lag2'] = 0
        combinations['lag3'] = 0
        combinations['lag6'] = 0
        combinations['lag12'] = 0
        combinations['lag24'] = 0
        combinations['lag36'] = 0
        combinations['datesold_year'] = pd.to_datetime(combinations['date']).dt.year
        combinations['month'] = pd.to_datetime(combinations['date']).dt.month

        combinations['price'] = 0

        combinations = pd.concat([dataset, combinations], axis=0)

        combinations['lag1'] = combinations.groupby(['postcode', 'propertyType', 'bedrooms'])['price'].shift(1)
        combinations['lag2'] = combinations.groupby(['postcode', 'propertyType', 'bedrooms'])['price'].shift(2)
        combinations['lag3'] = combinations.groupby(['postcode', 'propertyType', 'bedrooms'])['price'].shift(3)
        combinations['lag6'] = combinations.groupby(['postcode', 'propertyType', 'bedrooms'])['price'].shift(6)
        combinations['lag12'] = combinations.groupby(['postcode', 'propertyType', 'bedrooms'])['price'].shift(12)
        combinations['lag24'] = combinations.groupby(['postcode', 'propertyType', 'bedrooms'])['price'].shift(24)
        combinations['lag36'] = combinations.groupby(['postcode', 'propertyType', 'bedrooms'])['price'].shift(36)

        combinations = combinations.loc[combinations['date'] >= last_date]
        combinations = combinations.fillna(0)
        combinations = combinations.drop('date', axis=1)
        combinations = combinations[['postcode', 'propertyType', 'bedrooms',
                                    'datesold_year', 'month',
                                    'lag1', 'lag2', 'lag3', 'lag6', 'lag12', 'lag24', 'lag36']]
        return combinations

