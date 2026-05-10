import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
scaler = joblib.load('models/scaler.pkl')
rf_model = joblib.load('models/rf_model.pkl')
lr_model = joblib.load('models/lr_model.pkl')
ann_model = joblib.load('models/ann_model.pkl')
ann_model_2 = joblib.load('models/ann_model_2.pkl')
model_knr = joblib.load('models/model_knr.pkl')
model_lasso = joblib.load('models/model_lasso.pkl') 
model_GBR = joblib.load('models/model_GBR.pkl')
model_XGB = joblib.load('models/model_XGB.pkl')
rf_model_log = joblib.load('models/random_forest_logistic.pkl')


with open ('models/columns.json','r')as f:
    columns = json.load(f)

neighborhood_columns = [col for col in columns if col.startswith('neighborhood_id_')]

house_type_columns = [col for col in columns if col.startswith('house_type_id_')]
tab1,tab2 ,tab3,tab4,tab5,tab6= st.tabs(["Step 1","Step 2","Step 3","Step 4","Experiment","Experiment prediction"])

with tab1:
    st.title('Wybór atrybutów')
    st.header("Slisery dla wartośi numerycznych")

    slider_sq_mt_built = st.slider("SQ MT Built",13.0,900.0,13.0,10.0)
    st.write("Wybrano:",slider_sq_mt_built)

    slider_n_rooms = st.slider('Ilość Pokoje',1,24,1,1)
    st.write("Wybrano pokoje:",slider_n_rooms)

    slider_n_bathrooms = st.slider('Ilość łazienek',1,16,1,1)
    st.write("Wybrano łazienek:",slider_n_bathrooms)

    slider_floor = st.slider('Ilość piętr',1,10,1)
    st.write("Wybrano piętr:",slider_floor)

    slider_parking_price = st.slider("Cena za miejce parkingowe",0,600000,1000)
    st.write('Wybrana cena za miejsce parkingowe:',slider_parking_price)

    slider_built_year = st.slider('Rok budowy',1723,2022,1723,1)
    st.write('Rok:',slider_built_year)
with tab2:
    st.header('Checkboxy dla wartości True/False')

    has_ac = st.checkbox("Klimatyzator")

    has_lift = st.checkbox('Lift')

    has_pool = st.checkbox('Basen')

    has_terrace = st.checkbox('Terasa')

    has_balcony = st.checkbox('Balkon')

    is_renewal_needed = st.checkbox("Czy wymaga remontu?")

    has_central_heating = st.checkbox("Centralne Ogrzewanie")

    has_individual_heating = st.checkbox("Indywidualne Ogrzewanie")

    has_fitted_wardrobes = st.checkbox("Dopasowane szafy")

    is_exterior = st.checkbox('exterior')

    has_garden = st.checkbox("Ogród")

    has_storage_room = st.checkbox("pomieszczenie przechowywania") 

    is_accessible = st.checkbox('Dostępny')

    has_green_zones = st.checkbox('Strefy zielone')

    has_parking = st.checkbox("Parking")
    
    is_parking_included_in_price  = st.checkbox('Parking w cenie')

    is_orientation_north = st.checkbox('Orientacja północna')

    is_orientation_west = st.checkbox('Orientacja zachodnia')

    is_orientation_south = st.checkbox('Orientacja południowa') 

    is_orientation_east = st.checkbox('Orientacja wschodnia')
with tab3:

    st.header("Wybór okolice")

    neighborhood = st.selectbox('Okolice',neighborhood_columns)

    st.header("Wybór typu mieszkania")

    house_type = st.selectbox("Typ Mieszkania",house_type_columns)
with tab4:
    st.header("Predykcja ceny mieszkania")

    if st.button("Stworz dataset"):
        df = pd.DataFrame(columns=columns,data=np.zeros((1,len(columns))))
        df['sq_mt_built'] = slider_sq_mt_built

        df['n_rooms'] = slider_n_rooms

        df['n_bathrooms'] = slider_n_bathrooms

        df['floor'] = slider_floor

        df['parking_price'] = slider_parking_price

        df['built_year'] = slider_built_year

        df['has_ac'] = has_ac

        df['has_lift'] = has_lift

        df['has_pool'] = has_pool

        df['has_terrace'] = has_terrace

        df['has_balcony'] = has_balcony

        df['is_renewal_needed'] = is_renewal_needed

        df['has_central_heating'] = has_central_heating

        df['has_individual_heating'] = has_individual_heating

        df['has_fitted_wardrobes'] = has_fitted_wardrobes

        df['is_exterior'] = is_exterior

        df['has_garden'] = has_garden

        df['has_storage_room'] = has_storage_room

        df['is_accessible'] = is_accessible

        df['has_green_zones'] = has_green_zones

        df['has_parking'] = has_parking

        df['is_parking_included_in_price'] = is_parking_included_in_price

        df['is_orientation_north'] = is_orientation_north

        df['is_orientation_west'] = is_orientation_west

        df['is_orientation_south'] = is_orientation_south

        df['is_orientation_east'] = is_orientation_east

        df[neighborhood] = 1

        df[house_type] = 1

        df['Unnamed: 0'] = 0

        st.session_state['df'] = df

        st.dataframe(df)
    if st.button("Pokaż porównanie modeli"):
            df_scaled = scaler.transform(st.session_state['df'])

            rf_predict = rf_model.predict(df_scaled)

            lr_predict = lr_model.predict(df_scaled)

            ann_predict = ann_model.predict(df_scaled)

            ann_predict_2 = ann_model_2.predict(df_scaled)

            knr_predict = model_knr.predict(df_scaled)

            lasso_predict = model_lasso.predict(df_scaled)

            gbr_predict = model_GBR.predict(df_scaled)

            xgb_predict = model_XGB.predict(df_scaled)

            st.session_state['rf_predict'] = rf_predict[0]

            st.session_state['lr_predict'] = lr_predict[0]

            st.session_state["ann_predict"] = ann_predict[0]

            st.session_state["ann_predict_2"] = ann_predict_2[0]

            st.session_state['knr_predict'] = knr_predict[0]

            st.session_state['lasso_predict'] = lasso_predict[0]

            st.session_state['gbr_predict'] = gbr_predict[0]

            st.session_state['xgb_predict'] = xgb_predict[0]

            matrix1 = pd.DataFrame({
                "Random Forest:":f"{st.session_state.get('rf_predict', 'Brak predykcji'):,.0f} €",
                "Linear Regresion": [f"{st.session_state.get('lr_predict', 'Brak predykcji'):,.0f} €"],
                "ANN": [f"{st.session_state.get('ann_predict', 'Brak predykcji'):,.0f} €"],
                "ANN 2": [f"{st.session_state.get('ann_predict_2', 'Brak predykcji'):,.0f} €"],
                
                })
            
            matrix2 = pd.DataFrame({
                "KNeighborsRegressor:":f"{st.session_state.get('knr_predict', 'Brak predykcji'):,.0f} €",
                "Lasso": [f"{st.session_state.get('lasso_predict', 'Brak predykcji'):,.0f} €"],
                "Gradient Boosting Regressor": [f"{st.session_state.get('gbr_predict', 'Brak predykcji'):,.0f} €"],
                "XGBoost Regressor": [f"{st.session_state.get('xgb_predict', 'Brak predykcji'):,.0f} €"]
                })
            st.table(matrix1,border='horizontal')
            st.table(matrix2,border='horizontal')
    if st.button("Pokaz MAE error oraz RMSE error "):
         matrix_error = pd.DataFrame({
              "Model": ["Random Forest", "Linear Regression", "ANN", "ANN 2", "KNeighborsRegressor", "Lasso", "Gradient Boosting Regressor", "XGBoost Regressor"],
              "MAE Error":[110780.73, 195820.80,136807.18,157531.81,170705.42,195813.39,139990.02,142270.55 ],
              "RMSE Error":[248478.64, 347034.85, 389466.64, 324476.40,332617.86,347032.99,265012.33,272539.61]})
         matrix_error_sort = matrix_error.sort_values(by="MAE Error")
         st.table(matrix_error_sort,border='horizontal')

with tab5:
    st.title('Eksperyment z logarytmowaniem celu')
    st.text("Porównując poprzednie modele, można wyciągnąć wniosek, że model Random Forest ma najmniejsze MAE (110,780.73 euro), czyli średni błąd predykcji.")
    st.text('Najpierw wykorzystałem RandomizedSearchCV oraz GridSearchCV do znalezienia najlepszych hiperparametrów dla modelu Random Forest')
    st.text("Parametry:")
    
    code1 = '''parametesrs = {'n_estimators':[10,50,100,150,200],
    'max_features':['sqrt', 'log2'],
    'max_depth':[None, 10, 20, 30,40,50],
    "min_samples_split":[2,5,10],
    'min_samples_leaf':[1,2,4],
    'bootstrap':[True, False]
}'''
    st.code(code1)
    
    st.text('Teraz inicjalizacja Random Forest Regressor z parametrami')
    code2 = '''from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_model,param_distributions=parametesrs,n_iter=100,cv=5,verbose=2,random_state=42,n_jobs=1)
random_search.fit(X_train_scaled,y_train)
print("Best Parameters:", random_search.best_params_) '''
    st.code(code2)
    
    st.text('Prognozowanie:')
    code3 = ''' y_pred_updated = random_search.predict(X_test_scaled)'''
    st.code(code3)
    
    st.text('Obliczanie MAE:' )
    code4 = '''mae_error_new = mean_absolute_error(y_test, y_pred_updated)
print(f"MAE error: {mae_error_new:,.2f} euro")'''
    st.code(code4)
    
    st.text("Wynik:")
    st.code('MAE error: ' + f"{118441.72:,.2f}" + ' euro')
    
    st.text('Mamy gorzej, więc próbujemy GridSearchCV')
    code5 = '''from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=rf_model,param_grid=parametesrs,cv=5,verbose=2,n_jobs=1)
grid_search.fit(X_train_scaled,y_train)
print("Best Parameters:", grid_search.best_params_)'''
    st.code(code5)
    
    st.text('Najlepsze parametry:')
    code6 = ''' Best Parameters: {'bootstrap': False, 'max_depth': 50, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}'''   
    st.code(code6)
    
    st.text('Prognozowanie:')
    code7 = '''y_pred_updated_grid = grid_search.predict(X_test_scaled)''' 
    st.code(code7)
    
    st.text('Obliczanie MAE:') 
    code8 = '''mae_error_grid = mean_absolute_error(y_test, y_pred_updated_grid)'''
    st.code(code8)
    st.code('MAE error: ' + f"{117966.70:,.2f}" + ' euro')
    
    st.text('Mamy gorzej, więc próbujemy logarytmowania celu')
    st.title('Logarytmowanie celu')
    
    st.text('Najpierw logarytmowanie kolumny buy price')
    code9 = '''y_log = np.log1p(df['buy_price'])'''
    st.code(code9)
    
    st.text('Następnie podział na zbiór treningowy i testowy')
    code10 = '''X_train , X_test ,y_log_train,y_log_test = train_test_split(X,y_log,test_size=0.2,random_state=42)'''
    st.code(code10)
    
    st.text("Definiowanie i trenowanie modelu Random Forest Regressor z logarytmowanym celem")
    code11 = '''rf_log_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
rf_log_model.fit(X_train_scaled,y_log_train)'''
    st.code(code11)
    
    st.text('Prognozowanie:')  
    code12 = ''' y_rf_log_pred = rf_log_model.predict(X_test_scaled)'''
    st.code(code12)
    
    st.text('Obliczanie MAE i RMSE dla logarytmowanego celu:')
    code13 = '''rf_log_mae = mean_absolute_error(y_log_test, y_rf_log_pred)
rf_log_rmse = np.sqrt(mean_squared_error(y_log_test, y_rf_log_pred))'''
    st.code(code13)
    st.code('MAE error: 0.15 euro \nRMSE error: 0.22 euro')
    
    st.text('Teraz przekształcamy prognozy z powrotem do oryginalnej skali i obliczamy MAE i RMSE w euro:')
    code14 = '''rf_log_mae_2 = mean_absolute_error(np.expm1(y_log_test), np.expm1(y_rf_log_pred))
rf_log_rmse_2 = np.sqrt(mean_squared_error(np.expm1(y_log_test), np.expm1(y_rf_log_pred)))'''
    st.code(code14)
    st.code('MAE error: 109,208.08 euro \nRMSE error: 245,805.12 euro')
    
    st.subheader('Wnioski')
    st.write('Widzimy, że logarytmowanie celu poprawiło wyniki modelu Random Forest, zmniejszając MAE do około 109,208.08 euro i RMSE do około 245,805.12 euro, co jest najlepszym wynikiem, który udało nam się osiągnąć.')
with tab6:
     st.title('Predykcja z logarytmowanym modelem Random Forest')
     if st.button("Zrob predykcje z logarytmowanym modelem"):
          df_scaled = scaler.transform(st.session_state['df'])
          rf_model_log_predict = rf_model_log.predict(df_scaled)
          matrix3 = pd.DataFrame({
              "Random Forest Logarytmowany:":[f"{np.expm1(rf_model_log_predict[0]):,.0f} €"]
              })
          st.table(matrix3,border='horizontal')