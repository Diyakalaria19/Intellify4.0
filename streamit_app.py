"""
streamlit_app.py — main entry point. Pure UI orchestration: calls into
ollama_client.py (domain + goal extraction), specter2_encoder.py (embeddings),
router.py (folder loading), and search_engine.py (similarity + ranking).
No routing or similarity logic lives directly in this file.
"""

import streamlit as st

from auth import render_auth_ui
from runtime_domain_research_goal import extract_domain_and_goal
from specter2_encoder import encode as specter2_encode
from router import load_and_merge
from search_engine import run_search

st.set_page_config(page_title="Research Novelty Checker", layout="wide")


# Loading SPECTER2's model happens lazily inside specter2_encoder.py itself
# (module-level caching there), so nothing heavy needs @st.cache_resource here.
# We still warm it up once at the top so the first user query isn't the one
# that pays the model-load cost.
@st.cache_resource
def warm_up_specter2():
    specter2_encode("warmup")  # triggers model load once, result discarded
    return True


def main():
    if not render_auth_ui():
        return  # login/signup forms already rendered by auth.py

    warm_up_specter2()

    st.title("Research Novelty Checker")
    st.write("Enter your research idea or abstract to check how it compares against existing literature.")

    idea_description = st.text_area("Idea description / abstract", height=180)

    if st.button("Check Novelty", type="primary"):
        if not idea_description.strip():
            st.warning("Please enter a description first.")
            return

        with st.spinner("Extracting domain and research goal (Ollama/Gemma)..."):
            try:
                gpt_result = extract_domain_and_goal(idea_description)
            except Exception as e:
                st.error(
                    f"Could not reach the local Ollama server: {e}. "
                    f"Make sure Ollama is running (`ollama serve`) and the model is pulled."
                )
                return

        domains = gpt_result.get("domains", [])
        research_goal = gpt_result.get("research_goal", "")

        if not domains:
            st.error("Could not match your idea to a known domain. Try rephrasing.")
            return

        st.success(f"Detected domain(s): {', '.join(domains)}")
        st.caption(f"Extracted research goal: *{research_goal}*")

        with st.spinner("Loading category data and computing similarity..."):
            pool = load_and_merge(domains)
            if pool is None:
                st.error("No matching category data found for the detected domain(s).")
                return

            # both embeddings now go through SPECTER2, just on different text
            query_abstract_vec = specter2_encode(idea_description)
            query_goal_vec = specter2_encode(research_goal)

            try:
                results_df = run_search(query_abstract_vec, query_goal_vec, pool)
            except Exception as e:
                st.error(f"Something went wrong during search: {e}")
                return

        st.subheader("Top matching papers")

        for _, row in results_df.iterrows():
            with st.container(border=True):
                st.markdown(f"### {row['title']}")
                st.write(
                    f"**Overall similarity:** {row['final_score']:.2f}%  "
                    f"(abstract: {row['abstract_score']:.2f}% + goal: {row['goal_score']:.2f}%)"
                )
                st.write(row["abstract"][:400] + ("..." if len(row["abstract"]) > 400 else ""))

                if row.get("author"):
                    st.caption(f"Author(s): {row['author']}")

                arxiv_url = f"https://arxiv.org/abs/{row['paperid']}"
                st.markdown(f"[View on arXiv]({arxiv_url})")


if __name__ == "__main__":
    main()