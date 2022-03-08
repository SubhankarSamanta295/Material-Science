import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from sklearn.model_selection import train_test_split
import composition
from scipy import stats
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

def bulk_modulus_prediction(formula):
    df = pd.read_csv('Material Science.csv')

    uncleaned_formulae = df['ENTRY']

    cleaned_formulae = []

    for cell_value in uncleaned_formulae:
        split_list = cell_value.split(" [")
        clean_formula = split_list[0]
        cleaned_formulae.append(clean_formula)

    df_cleaned = pd.DataFrame()
    df_cleaned['formula'] = cleaned_formulae
    df_cleaned['bulk_modulus'] = df['AEL VRH bulk modulus']

    check_for_duplicates = df_cleaned['formula'].value_counts()
    df_cleaned.drop_duplicates('formula', keep='first', inplace=True)

    df_cleaned.columns = ['formula', 'target']

    X, y, formulae = composition.generate_features(df_cleaned)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=25)

    def handling_correlation(X_train,threshold):
        corr_features = set()
        corr_matrix = X_train.corr()
        for i in range(len(corr_matrix .columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) >threshold:
                    colname = corr_matrix.columns[i]
                    corr_features.add(colname)
        return list(corr_features)

    train=X_train.copy()
    dropped = handling_correlation(train.copy(),0.85)

    X_train.drop(dropped,axis=1,inplace=True)
    X_test.drop(dropped,axis=1,inplace=True)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)

    xgb_regressor = xgb.XGBRegressor(learning_rate=0.05, n_estimators=520, subsample=0.65,n_jobs=7,max_depth=7,random_state=25)
    xgb_regressor.fit(X_train,y_train)

    class MaterialsModel():
        def __init__(self, trained_model, sc):
            self.model = trained_model
            self.scalar = sc
        
        def predict(self, formula):
            if type(formula) is str:
                df_formula = pd.DataFrame()
                df_formula['formula'] = [formula]
                df_formula['target'] = [0]
            if type(formula) is list:
                df_formula = pd.DataFrame()
                df_formula['formula'] = formula
                df_formula['target'] = np.zeros(len(formula))
            # here we get the features associated with the formula
            X, y, formula = composition.generate_features(df_formula)
            # here we scale the data (acording to the training set statistics)
            X.drop(dropped,axis=1,inplace=True)
            
            X_scaled = self.scalar.transform(X)
            y_predicted = self.model.predict(X_scaled)
            # save our predictions to a dataframe
            prediction = pd.DataFrame(formula)
            prediction['predicted value'] = y_predicted
            return prediction

    # initialize an object to hold our bulk modulus model
    bulk_modulus_model = MaterialsModel(xgb_regressor, sc)
    #formulae_to_predict = ['Ag1Al1S2']
    # use the bulk modulus object to generate predictions for our formulae!
    bulk_modulus_prediction = bulk_modulus_model.predict([formula])

    return bulk_modulus_prediction['predicted value'][0]