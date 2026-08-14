"""
auth.py — signup/login logic for Streamlit, backed by SQLite (app.db).

Requires: pip install bcrypt streamlit
"""

import bcrypt
import streamlit as st

from db_setup import get_connection, init_db


def hash_password(plain_password: str) -> str:
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, stored_hash: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), stored_hash.encode("utf-8"))


def signup(username: str, email: str, password: str) -> tuple[bool, str]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
    if cursor.fetchone():
        conn.close()
        return False, "Username or email already exists."

    password_hash = hash_password(password)
    cursor.execute(
        "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
        (username, email, password_hash),
    )
    conn.commit()
    conn.close()
    return True, "Account created successfully. Please log in."


def login(username: str, password: str) -> tuple[bool, str]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    if not row:
        return False, "No account found with that username."

    if not verify_password(password, row["password_hash"]):
        return False, "Incorrect password."

    st.session_state["logged_in"] = True
    st.session_state["username"] = row["username"]
    st.session_state["user_id"] = row["id"]

    return True, f"Welcome back, {row['username']}!"


def logout():
    st.session_state["logged_in"] = False
    st.session_state.pop("username", None)
    st.session_state.pop("user_id", None)


def render_auth_ui():
    init_db()  # safe no-op if tables already exist; ensures app.db is ready on first-ever run

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]:
        st.sidebar.write(f"Logged in as **{st.session_state['username']}**")
        if st.sidebar.button("Log out"):
            logout()
            st.rerun()
        return True

    st.title("Log in or sign up")
    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In")
            if submitted:
                success, message = login(username, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose a password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if len(new_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    success, message = signup(new_username, new_email, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

    return False