
import logging
import io
import streamlit as st


def main():

    # Title of the page
    st.title("Skin cancer recognition")

    # Display a text
    st.write("This all uses machine learning model to detect skin cancer based on provided image")

    uploaded_file_data = st.file_uploader("Import photo of your skin issue", type=["jpg", "jpge", "png"], accept_multiple_files=False)


    if uploaded_file_data is not None:
        file_details = {
            "File name": uploaded_file_data.name,
            "File type": uploaded_file_data.type,
            "File size": uploaded_file_data.size,
        }

        # Display file details
        st.write("### File Details")
        st.json(file_details)

        bytes_data = uploaded_file_data.read()
        st.write("filename:", uploaded_file_data.name)
        st.write(bytes_data)



if __name__ == '__main__':
    logging.debug("Starting python...")
    main()


