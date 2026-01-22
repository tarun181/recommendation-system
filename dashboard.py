import streamlit as st
import requests

# 1. Configuration: Replace with your actual Render API URL
# IMPORTANT: No trailing slash "/" at the end
API_URL = "https://recommendation-system-ojx3.onrender.com"

# 2. App Title & Styling
st.set_page_config(page_title="Amazon Recommender", page_icon="🛒", layout="wide")

st.title("🛒 AI-Powered Hybrid Recommender System")
st.markdown("""
    **Architecture:** LightGBM Ranking + ALS Retrieval  
    **Goal:** Personalized Product Recommendations based on User Purchase History.
""")
st.divider()

# 3. Sidebar for Inputs
with st.sidebar:
    st.header("🔍 User Controls")
    user_id = st.number_input("User ID", min_value=0, value=4, step=1, help="Enter a User ID (e.g. 4, 110)")
    top_k = st.slider("Number of Recommendations", min_value=1, max_value=8, value=4)
    st.info("💡 Try User IDs: 4, 110, 25")

    if st.button("Get Recommendations", type="primary"):
        fetch_data = True
    else:
        fetch_data = False

# 4. Main Logic
if fetch_data:
    with st.spinner("🤖 Models are crunching numbers..."):
        try:
            # Call your FASTAPI backend
            payload = {"user_idx": user_id, "top_k": top_k}
            response = requests.post(f"{API_URL}/recommend", json=payload)
            response.raise_for_status()  # Check for errors (404, 500)

            data = response.json()
            recs = data["recommendations"]

            st.success(f"✅ Found {len(recs)} personalized items for User {user_id}")

            # 5. Display Cards in a Grid
            # Calculate rows needed
            cols = st.columns(4)  # 4 items per row

            for i, item in enumerate(recs):
                with cols[i % 4]:  # Wrap around columns
                    # --- THE MAGIC PART ---
                    asin = item['asin']

                    # Amazon Image URL Hack
                    image_url = f"http://images.amazon.com/images/P/{asin}.01._SS200_.jpg"

                    # Card Container
                    with st.container(border=True):
                        # Display Image (with fallback)
                        try:
                            st.image(image_url, use_container_width=True)
                        except:
                            st.image("https://via.placeholder.com/150", caption="No Image")

                        st.caption(f"ASIN: {asin}")

                        # Clickable Button
                        st.link_button("👉 View on Amazon", item['amazon_link'])

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.warning("Make sure your Render Backend is live!")

else:
    st.info("👈 Click 'Get Recommendations' to see the model in action!")