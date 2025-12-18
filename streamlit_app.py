import streamlit as st
import requests

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_URL = "https://shl-assessment-api-fz0n.onrender.com/recommend"

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📊",
    layout="centered"
)

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("📊 SHL Assessment Recommendation System")

st.write(
    """
    Enter a job description or role-based query to receive
    the most relevant SHL **individual assessments**.
    """
)

query = st.text_area(
    "Job Query / Job Description",
    placeholder="e.g. Entry level investment banking professional",
    height=120
)

top_k = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

# -------------------------------------------------
# ACTION
# -------------------------------------------------
if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query before submitting.")
    else:
        with st.spinner("Generating recommendations..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": query,
                        "top_k": top_k
                    },
                    timeout=60
                )

                if response.status_code != 200:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                else:
                    results = response.json()

                    if not results:
                        st.info("No relevant assessments found.")
                    else:
                        st.success("Recommendations generated successfully!")

                        for r in results:
                            st.markdown(f"### Rank {r['rank']}")
                            st.markdown(f"**Assessment URL:** {r['url']}")
                            st.markdown(f"**Reasoning:** {r['reasoning']}")
                            st.markdown("---")

            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again.")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
