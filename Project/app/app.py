import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import gc 
from sklearn.metrics import mean_absolute_error, mean_squared_error

@st.cache_resource
def load_light_models_and_data():
    scaler = joblib.load('models/scaler.pkl')
    lr_model = joblib.load('models/lr_model.pkl')
    ann_model = joblib.load('models/ann_model.pkl')
    ann_model_2 = joblib.load('models/ann_model_2.pkl')
    model_lasso = joblib.load('models/model_lasso.pkl') 
    model_GBR = joblib.load('models/model_GBR.pkl')
    model_XGB = joblib.load('models/model_XGB.pkl')
    
    with open('models/columns.json', 'r') as f:
        columns = json.load(f)
        
    return scaler, lr_model, ann_model, ann_model_2, model_lasso, model_GBR, model_XGB, columns

scaler, lr_model, ann_model, ann_model_2, model_lasso, model_GBR, model_XGB, columns = load_light_models_and_data()
# -------------------------------------------------------------------

neighborhood_columns = [col for col in columns if col.startswith('neighborhood_id_')]
house_type_columns = [col for col in columns if col.startswith('house_type_id_')]

tab1, tab2, tab3, tab4 = st.tabs(["Step 1", "Step 2", "Step 3", "Step 4"])

with tab1:
    st.title('Wybór atrybutów')
    st.header("Slisery dla wartośi numerycznych")

    slider_sq_mt_built = st.slider("SQ MT Built", 13.0, 900.0, 13.0, 10.0)
    st.write("Wybrano:", slider_sq_mt_built)

    slider_n_rooms = st.slider('Ilość Pokoje', 1, 24, 1, 1)
    st.write("Wybrano pokoje:", slider_n_rooms)

    slider_n_bathrooms = st.slider('Ilość łazienek', 1, 16, 1, 1)
    st.write("Wybrano łazienek:", slider_n_bathrooms)

    slider_floor = st.slider('Ilość piętr', 1, 10, 1)
    st.write("Wybrano piętr:", slider_floor)

    slider_parking_price = st.slider("Cena za miejce parkingowe", 0, 600000, 1000)
    st.write('Wybrana cena za miejsce parkingowe:', slider_parking_price)

    slider_built_year = st.slider('Rok budowy', 1723, 2022, 1723, 1)
    st.write('Rok:', slider_built_year)

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
    is_exterior = st.checkbox('Exterior')
    has_garden = st.checkbox("Ogród")
    has_storage_room = st.checkbox("Pomieszczenie przechowywania") 
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
    neighborhood = st.selectbox('Okolice', neighborhood_columns)

    st.header("Wybór typu mieszkania")
    house_type = st.selectbox("Typ Mieszkania", house_type_columns)

with tab4:
    st.header("Predykcja ceny mieszkania")

    if st.button("Stworz dataset"):
        df = pd.DataFrame(columns=columns, data=np.zeros((1, len(columns))))
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
        # Używamy spinnera, ponieważ ładowanie modeli z dysku zajmie chwilę
        with st.spinner("Uwaga: Obliczanie ciężkich modeli... (może to zająć od 10 do 60 sekund)"):
            df_scaled = scaler.transform(st.session_state['df'])

            # 1. Predykcje LEKKICH modeli (już są w RAM, więc to ułamek sekundy)
            st.session_state['lr_predict'] = lr_model.predict(df_scaled)[0]
            st.session_state["ann_predict"] = ann_model.predict(df_scaled)[0]
            st.session_state["ann_predict_2"] = ann_model_2.predict(df_scaled)[0]
            st.session_state['lasso_predict'] = model_lasso.predict(df_scaled)[0]
            st.session_state['gbr_predict'] = model_GBR.predict(df_scaled)[0]
            st.session_state['xgb_predict'] = model_XGB.predict(df_scaled)[0]

            rf_temp = joblib.load('models/rf_model.joblib.gz')
            st.session_state['rf_predict'] = rf_temp.predict(df_scaled)[0]
            del rf_temp  
            gc.collect() 

            knr_temp = joblib.load('models/model_knr.pkl')
            st.session_state['knr_predict'] = knr_temp.predict(df_scaled)[0]
            del knr_temp
            gc.collect()

        matrix1 = pd.DataFrame({
            "Random Forest:": f"{st.session_state.get('rf_predict', 'Brak predykcji'):,.0f} €",
            "Linear Regresion": [f"{st.session_state.get('lr_predict', 'Brak predykcji'):,.0f} €"],
            "ANN": [f"{st.session_state.get('ann_predict', 'Brak predykcji'):,.0f} €"],
            "ANN 2": [f"{st.session_state.get('ann_predict_2', 'Brak predykcji'):,.0f} €"],
        })
            
        matrix2 = pd.DataFrame({
            "KNeighborsRegressor:": f"{st.session_state.get('knr_predict', 'Brak predykcji'):,.0f} €",
            "Lasso": [f"{st.session_state.get('lasso_predict', 'Brak predykcji'):,.0f} €"],
            "Gradient Boosting Regressor": [f"{st.session_state.get('gbr_predict', 'Brak predykcji'):,.0f} €"],
            "XGBoost Regressor": [f"{st.session_state.get('xgb_predict', 'Brak predykcji'):,.0f} €"]
        })
        
        st.success("Obliczenia zakończone sukcesem!")
        st.table(matrix1)
        st.table(matrix2)

    if st.button("Pokaz MAE error oraz RMSE error "):
         matrix_error = pd.DataFrame({
              "Model": ["Random Forest", "Linear Regression", "ANN", "ANN 2", "KNeighborsRegressor", "Lasso", "Gradient Boosting Regressor", "XGBoost Regressor"],
              "MAE Error": [110780.73, 195820.80, 136807.18, 157531.81, 170705.42, 195813.39, 139990.02, 142270.55],
              "RMSE Error": [248478.64, 347034.85, 389466.64, 324476.40, 332617.86, 347032.99, 265012.33, 272539.61]
         })
         matrix_error_sort = matrix_error.sort_values(by="MAE Error")
         st.table(matrix_error_sort)