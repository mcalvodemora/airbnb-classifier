import numpy as np
import torch
import torchvision
from PIL import Image
import streamlit as st
from matplotlib import pyplot as plt
from torchvision import transforms
from cnn.cnn import CNN, load_model_weights
from training_models import load_training_data
import torch.nn.functional as F


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.squeeze().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    fig = plt.figure()
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    return fig


def load_image(image, img_size):
    #  Función para cargar y transformar los datos de entrenamiento y validación.
    #  Aplica transformaciones de aumento de datos como rotación y recorte aleatorio para el conjunto de entrenamiento.

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize to the size expected by your model
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    image_tensor = transform(image)
    image_tensor = image_tensor.expand(3, -1, -1)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def predict_(image_tensor, model, name_saving_model, device):
    model_weights = load_model_weights(name_saving_model)
    my_trained_model = model
    my_trained_model.load_state_dict(model_weights)
    my_trained_model.eval()
    image_tensor = image_tensor.to(device)
    outputs = my_trained_model(image_tensor)
    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs, dim=1)

    # Get top n predictions and their probabilities
    top_n_probabilities, top_n_indices = torch.topk(probabilities, 2, dim=1)

    # Convert tensor to numpy arrays
    top_n_probabilities = top_n_probabilities.cpu().detach().numpy().flatten()
    top_n_indices = top_n_indices.cpu().detach().numpy().flatten()

    return top_n_indices, top_n_probabilities


def show_results_(image_tensor, top_n_indices, top_n_probabilities, valid_loader):
    classnames = valid_loader.dataset.classes
    labels = [classnames[idx] for idx in top_n_indices]
    fig = imshow(image_tensor, title=f"{labels[0]}")
    st.pyplot(fig)

def main():

    # -------------
    # -- Params. --
    # -------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar el modelo entrenado y los datos de carga de imágenes

    train_loader, valid_loader, num_classes = load_training_data(64)

    model = CNN(torchvision.models.resnet50(weights='DEFAULT'), 15, unfreezed_layers=7,
                dropout=0.2).to(device)
    name_saving_model = "resnet50-30epoch-0.0001lr-64bs-0.2do-7unlay"

    # -----------------------------------------------------------------------------------------------
    # ------------------------- STREAMLIT APP CONFIGURATION: --------------------------
    # -----------------------------------------------------------------------------------------------

    # ---------------------
    # ---- Diseño app -----
    # ---------------------
    
    # Configuración de la aplicación
    # Configuración de la página por defecto
    st.set_page_config(layout="wide")
    
    st.markdown(
        """
        <style>
            body {
                background-color: black;
            }
        </style>
        """,
        unsafe_allow_html=True
    )



    # Estilo del fondo de la página
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f0f0;
            }
            .stApp {
                max-width: 3200px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    # Título de la app
    st.markdown(
        """
        <div style='background-color: #f63366; padding: 20px; border-radius: 10px; margin-bottom: 20px; font-family: 'Roboto', sans-serif;'>
            <h1 style='color: white; text-align: center; font-size: 36px;'>Clasificación de Imágenes para Airbnb HouseIdentifyAI</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Contenedor para la etiqueta de texto en la parte superior
    st.markdown(
        """
        <div style='background-color: grey; padding: 3px; border-radius: 5px; margin-bottom: 10px;'>
            <h5 style='color: white; text-align: center;'>¡Bienvenido a la aplicación de clasificación de imágenes HouseIdentifyAI!</h5>
        </div>
        """,
        unsafe_allow_html=True
    )
    
        
    # Logo
    logo = Image.open("logo.png")
    #st.image(logo, use_column_width=False, width=250, caption="", clamp=True)


    # Panel lateral a la izquierda
    st.sidebar.header("Acerca HouseIdentifyAI")
    st.sidebar.image(logo, use_column_width=False, width=250, caption="Logo de HouseIdentifyAI")
    st.sidebar.markdown("Somos una App clasificadora de imágenes de viviendas y alrededores. Nuestra solución ofrece un valor significativo al proporcionar una herramienta fácil de usar que simplifica el proceso de clasificación de imágenes para los propietarios de propiedades en Airbnb.")
    

    # Panel lateral a la derecha
    st.sidebar.empty()

    # Contenedor principal para la página
    st.subheader('Instrucciones:')
    st.markdown(
        """
        <div style='background-color: #ffcc00; padding: 10px; border-radius: 5px; color: black; font-family: 'Roboto', sans-serif; font-size: 26px;'>
            <p>1. Carga la imagen deseada.</p>
            <p>2. Haz clic en el botón 'Clasificar'.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Cargar imagen desde el usuario
    st.subheader('Selecciona una imagen:')
    uploaded_file = st.file_uploader(
    "",
    type=['jpg', 'png'],
    key="fileUploader",
    accept_multiple_files=False)

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Imagen cargada', use_column_width=True)

        # Botón de clasificación
        if st.button('Clasificar'):
            # Convertir la imagen cargada a un tensor y realizar la clasificación
            image = Image.open(uploaded_file).convert('L')

            image_tensor = load_image(image, img_size=224)
            top_n_indices, top_n_probabilities = predict_(image_tensor, model, name_saving_model, device)


            # Mostrar el resultado de la clasificación
            show_results_(image_tensor, top_n_indices, top_n_probabilities, valid_loader)
            st.success('La imagen ha sido clasificada.')

    


if __name__ == "__main__":
    main()