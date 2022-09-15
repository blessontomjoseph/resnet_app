import streamlit as st
import torch
from PIL import Image

model=torch.load('model.pt')
metadata=torch.load('metadata')
labels = metadata['classes']
trans = metadata['transformations']

if __name__ == '__main__':

    image=st.file_uploader('get a label')
    apply = st.button('apply')
        
    if apply:
        st.image(image)
        example=Image.open(image)
        batch = trans(example).unsqueeze(0)
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = labels[class_id]
        st.write(f"{category_name}: {100 * score:.1f}%")   
    
    

