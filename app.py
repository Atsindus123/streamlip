# import streamlit as st
# import pandas as pd
# import pickle

# # Load the model
# with open('car.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# # Define function to make prediction
# def predict_price(features):
#     # Preprocess the input features (if needed)
#     # Here you would need to preprocess your input features to match the format used during training
    
#     # Make prediction
#     prediction = model.predict(features)
#     return prediction

# # Streamlit interface
# def main():
#     st.title("Car Price Prediction")

#     # Input features
#     name = st.text_input("Car Name")
#     location_options = ['Ahmedabad', 'Bangalore', 'Chennai', 'Coimbatore', 'Delhi',
#         'Hyderabad', 'Jaipur', 'Kochi', 'Kolkata', 'Mumbai', 'Pune']
#     location = st.selectbox("Location", location_options)
#     year = st.slider("Year", 2000, 2022, 2010)
#     kilometers_driven = st.number_input("Kilometers Driven", value=10000)
#     fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
#     transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
#     owner_type = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
#     mileage = st.number_input("Mileage (kmpl)", value=15.0)
#     engine = st.number_input("Engine (CC)", value=1000)
#     power = st.number_input("Power (bhp)", value=60)
#     seats = st.number_input("Seats", value=5)

#     # Predict button
#     if st.button("Predict"):
#         features = [[name, location, year, kilometers_driven, fuel_type, transmission, owner_type, mileage, engine, power, seats]]
#         prediction = predict_price(features)
#         st.success(f"Predicted Price: {prediction}")

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('csk.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define function to make prediction
def predict_price(features):
    try:
        # Make prediction
        prediction = model.predict(features)
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Streamlit interface
def main():
    st.title("Car Price Prediction")

    # Input features
    name = st.text_input("Car Name")
    location_options = ['Ahmedabad', 'Bangalore', 'Chennai', 'Coimbatore', 'Delhi',
        'Hyderabad', 'Jaipur', 'Kochi', 'Kolkata', 'Mumbai', 'Pune']
    location = st.selectbox("Location", location_options)
    year = st.slider("Year", 2000, 2022, 2010)
    kilometers_driven = st.number_input("Kilometers Driven", value=10000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner_type = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
    mileage = st.number_input("Mileage (kmpl)", value=15.0)
    engine = st.number_input("Engine (CC)", value=1000)
    power = st.number_input("Power (bhp)", value=60)
    seats = st.number_input("Seats", value=5)

    # Predict button
    if st.button("Predict"):
        # Convert input features to DataFrame
        features_df = pd.DataFrame({
            'Name': [name],
            'Location': [location],
            'Year': [year],
            'Kilometers_Driven': [kilometers_driven],
            'Fuel_Type': [fuel_type],
            'Transmission': [transmission],
            'Owner_Type': [owner_type],
            'Mileage': [mileage],
            'Engine': [engine],
            'Power': [power],
            'Seats': [seats]
        })

        # Make sure all columns are numeric
        features_df['Year'] = features_df['Year'].astype(int)
        features_df['Kilometers_Driven'] = features_df['Kilometers_Driven'].astype(int)
        features_df['Mileage'] = features_df['Mileage'].astype(float)
        features_df['Engine'] = features_df['Engine'].astype(int)
        features_df['Power'] = features_df['Power'].astype(float)
        features_df['Seats'] = features_df['Seats'].astype(int)

        # Handle missing values
        features_df.fillna(0, inplace=True)  # Replace NaN values with 0

        # Make prediction
        prediction = predict_price(features_df)
        if prediction is not None:
            st.success(f"Predicted Price: {prediction}")

if __name__ == "__main__":
    main()
# 


# import streamlit as st
# import pandas as pd
# import pickle 

# # Load the trained model from the .pkl file
# with open('car.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Define allowed car brands
# allowed_brands = ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
#                   'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
#                   'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche', 'renault',
#                   'saab', 'subaru', 'toyota', 'volkswagen', 'volvo']

# # Define prediction function
# def predict_price(Name, Location, Year, Kilometers_Driven, Fuel_Type, Transmission, Owner_Type, Mileage, Engine, Power,Seats):
#     # Check if brand is in the allowed brands
#     if brand.lower() not in allowed_brands:
#         return None, f"Please select a brand from the following list: {', '.join(allowed_brands)}"

#     # Check if any of the input fields are empty
#     if any(param == "" for param in [fuel_type, type_design, engine_location, engine_type]):
#         return None, "Please fill in all the input fields."

#     # Check if any of the input features is 0.00
#     if any(val == 0.00 for val in [engine_size, horse_power, top_rpm, city_mileage, highway_mileage]):
#         return None, "Please ensure all input features are greater than 0.00."

#     # Create a DataFrame with the input features
#     input_data = pd.DataFrame({
#         'Brand Name': [brand],
#         'Fuel type': [fuel_type],
#         'Design': [type_design],
#         'Engine Location': [engine_location],
#         'Engine Type': [engine_type],
#         'Engine Size': [engine_size],
#         'Horse Power': [horse_power],
#         'Top-RPM': [top_rpm],
#         'City Mileage': [city_mileage],
#         'Highway Mileage': [highway_mileage]
#     })
    
#     # Use the loaded model to make predictions
#     predicted_price = model.predict(input_data)[0]  # Assuming your model returns a single prediction
    
#     return predicted_price, None

# # Streamlit app
# st.title('ATS Car Price Predictor')

# # Add background image using HTML
# st.markdown(
#     """
#     <style>
#     body {
#         background-image: url("https://www.hdcarwallpapers.com/walls/aston_martin_dbs_770_ultimate_2023_8k_3-HD.jpg");
#         background-size: cover;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Add input widgets with improved styling
# with st.sidebar.form(key='input_form'):
#     st.sidebar.header('Input Features')
#     brand = st.selectbox('Brand Name', allowed_brands)
#     fuel_type = st.selectbox('Fuel Type', ['Diesel', 'Petrol', 'Gas'])
#     type_design = st.selectbox('Type Design', ['Convertible', 'Hardtop', 'Hatchback', 'Sedan', 'Wagon'])
#     engine_location = st.selectbox('Engine Location', ['Front', 'Rear'])
#     engine_type = st.selectbox('Engine Type', ['DOHC', 'DOHCV', 'L', 'OHC', 'OHCF', 'OHCV', 'Rotor'])
#     engine_size = st.number_input('Engine Size (cc)', min_value=0.01, format="%.2f")
#     horse_power = st.number_input('Horse Power', min_value=0.01, format="%.2f")
#     top_rpm = st.number_input('Top-RPM', min_value=0.01, format="%.2f")
#     city_mileage = st.number_input('City Mileage (mpg)', min_value=0.01, format="%.2f")
#     highway_mileage = st.number_input('Highway Mileage (mpg)', min_value=0.01, format="%.2f")

#     predict_button = st.form_submit_button(label='Predict')

# # Add a section to display predicted price
# if predict_button:
#     predicted_price, error_message = predict_price(brand, fuel_type, type_design, engine_location, engine_type, engine_size, horse_power, top_rpm, city_mileage, highway_mileage)
#     if error_message:
#         st.error(error_message)
#     elif predicted_price is not None:
#         st.success(f'Predicted Price: ${predicted_price:.2f}')

# elif all([brand, fuel_type, type_design, engine_location, engine_type, engine_size, horse_power, top_rpm, city_mileage, highway_mileage]):
#     predicted_price, _ = predict_price(brand, fuel_type, type_design, engine_location, engine_type, engine_size, horse_power, top_rpm, city_mileage, highway_mileage)
#     st.write(f'Fill all the input fields to view the predicted price: {predicted_price:.2f}')
