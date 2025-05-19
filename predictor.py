import pandas as pd

def predict_rating(restaurant_data, model_artifacts):
    model = model_artifacts['model']
    scaler_selection = model_artifacts['scaler_selection']
    encoders = model_artifacts['encoders']
    mlb = model_artifacts['cuisine_mlb']
    selected_features = model_artifacts['selected_features']

    data = {}
    for col, encoder in encoders.items():
        if col in restaurant_data:
            try:
                data[col] = encoder.transform([restaurant_data[col]])[0]
            except:
                data[col] = 0
        else:
            data[col] = 0 

    cuisine_features = {}
    if 'Cuisines' in restaurant_data:
        cuisine_list = [c.strip() for c in restaurant_data['Cuisines'].split(',')]
        cuisine_array = mlb.transform([cuisine_list])[0]
        for i, cuisine in enumerate(mlb.classes_):
            cuisine_features[cuisine] = cuisine_array[i]

    for col in restaurant_data:
        if col not in encoders and col != 'Cuisines':
            data[col] = restaurant_data[col]

    combined_data = {**data, **cuisine_features}
    input_df = pd.DataFrame([combined_data])

    expected_columns = scaler_selection.feature_names_in_
    input_df_aligned = pd.DataFrame(0, index=[0], columns=expected_columns)

    for col in input_df.columns:
        if col in expected_columns:
            input_df_aligned[col] = input_df[col]

    scaled_input = scaler_selection.transform(input_df_aligned)
    input_df_scaled = pd.DataFrame(scaled_input, columns=expected_columns)

    final_input = input_df_scaled[selected_features]

    rating = model.predict(final_input)[0]
    return rating