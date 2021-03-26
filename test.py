import time
import streamlit as st
from robustnessgym import Dataset


@st.cache
def load_dataset(path: str):
    time.sleep(10)
    return Dataset.load_from_disk(path)


if __name__ == '__main__':
    dataset = load_dataset('preprocessing/cnn_dailymail_v3')
    index = st.slider(label='A', min_value=0, max_value=len(dataset))
    st.write(dataset[index])