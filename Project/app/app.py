import streamlit as st
import joblib
import json
scaler = joblib.load('models/scaler.pkl')
rf_model = joblib.load('models/rf_model.pkl')
lr_model = joblib.load('models/lr_model.pkl')
ann_model = joblib.load('models/ann_model.pkl')
ann_model_2 = joblib.load('models/ann_model_2.pkl')

with open ('models/columns.json','r')as f:
    columns = json.load(f)

neighborhood_columns = [col for col in columns if col.startswith('neighborhood_id_')]

house_type_columns = [col for col in columns if col.startswith('house_type_id_')]
tab1,tab2 ,tab3= st.tabs(["Step 1","Step 2","Step 3"])

with tab1:
    st.title('Wybór atrybutów')
    st.header("Slisery dla wartośi numerycznych")

    slider_sq_mt_built = st.slider("SQ MT Built",13.0,900.0,13.0,10.0)
    st.write("Wybrano:",slider_sq_mt_built)

    slider_n_rooms = st.slider('Ilość Pokoje',1,24,1,1)
    st.write("Wybrano pokoje:",slider_n_rooms)

    slider_n_bathrooms = st.slider('Ilość łazienek',1,16,1,1)
    st.write("Wybrano łazienek:",slider_n_bathrooms)

    slider_floor = st.slider('Ilość piętr',0,10,1)
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