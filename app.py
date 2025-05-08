import streamlit as st
import pickle
import pandas as pd
import streamlit_authenticator as stauth

# ----- USER AUTHENTICATION SETUP -----
names = ['John Doe']
usernames = ['johndoe']
passwords = ['12345']  # plain text, for demo only

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(
    {"usernames": {
        usernames[0]: {
            "name": names[0],
            "password": hashed_passwords[0]
        }
    }},
    "myapp", "abcdef", cookie_expiry_days=1
)

name, auth_status, username = authenticator.login('Login', 'main')

if auth_status:
    st.sidebar.success(f"Welcome, {name} 👋")
    authenticator.logout('Logout', 'sidebar')

    # Load the model and encoder
    with open("xg_boost.pkl", "rb") as f:
        bundle = pickle.load(f)
        model = bundle["model"]
        encoder = bundle["encoder"]
        selected_features = bundle["selected_features"]

    # Streamlit UI
    st.title("📞 Call Center Service Outcome Predictor")

    st.markdown("### Enter Call Details:")

    # Categorical inputs
    sector = st.selectbox("Sector", ["Finance", "Healthcare", "Retail", "Telecom"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])
    complaint_type = st.selectbox("Complaint Type", ["Billing", "Technical", "General"])

    # Numeric inputs
    call_duration = st.number_input("Call Duration (in minutes)", step=1, min_value=0)
    customer_age = st.number_input("Customer Age", step=1, min_value=10)
    operator_id = st.number_input("Operator ID", step=1, min_value=1)
    satisfaction_score = st.slider("Satisfaction Score", 0.0, 1.0, step=0.01)

    if st.button("Predict Outcome"):
        try:
            # Prepare input data
            categorical_data = {
                "Sector": sector,
                "Region": region,
                "Day_Type": day_type,
                "Complaint_Type": complaint_type
            }

            numeric_data = {
                "Call_Duration": call_duration,
                "Customer_Age": customer_age,
                "Operator_ID": operator_id,
                "Satisfaction_Score": satisfaction_score
            }

            cat_df = pd.DataFrame([categorical_data])
            num_df = pd.DataFrame([numeric_data])

            # Encode categorical columns
            encoded_cat = encoder.transform(cat_df)
            encoded_cat_df = pd.DataFrame(
                encoded_cat.toarray(),
                columns=encoder.get_feature_names_out()
            )

            # Combine and select final input
            final_input = pd.concat([encoded_cat_df, num_df], axis=1)
            final_input = final_input[selected_features]

            # Predict
            prediction = model.predict(final_input)
            st.success(f"📈 Predicted Outcome: {prediction[0]}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

elif auth_status is False:
    st.error("Username or password is incorrect")

elif auth_status is None:
    st.warning("Please enter your username and password")
