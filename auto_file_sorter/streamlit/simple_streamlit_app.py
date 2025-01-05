import streamlit as st

# TODO: Create a simple streamlit app to accept a text prompt and them pydantic_ai with dataclasses for deps_type and BaseModel for result_type  for information that we return from the sql db about what we have stored for the user and also to add new rules if they ask for them.
# TODO : this run code here would need to be run as part of the process of spinning up the GmailAutomation daemon.
# The streamlit app will need to show a log of the prompts, responses and all the logs from the application like the GmailAutomation daemon.
# It will also need to allow me to interact with the gmail db and the rules db.
# TODO: I will then need to add all this functionality to the react js front end app in root/../auto-file-sorter-react-app/src/mains/streamlit_app.tsx


def main():
    # Set page title
    st.title("My Simple Streamlit App")

    # Add some text
    st.write("Welcome to my first Streamlit app!")

    # Add a text input
    user_input = st.text_input("Enter your name:")

    # Add a button
    if st.button("Say Hello"):
        if user_input:
            st.write(f"Hello, {user_input}! 👋")
        else:
            st.write("Please enter your name!")

    # Add a slider
    number = st.slider("Pick a number", 0, 100)
    st.write(f"Your number is: {number}")


if __name__ == "__main__":
    main()
