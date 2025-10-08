import rootutils
import numpy as np
from tqdm import tqdm

root_path = rootutils.setup_root(".", indicator=".project-root", pythonpath=True)


from src.data_loader import load_lfw_dataset
from src.face_processor import FaceProcessor



def run():
    """
    Function to run the LFW processing script.
    """
    print("--- STARTING LFW PROCESSING SCRIPT ---")
    

    processor = FaceProcessor()

    images, labels, target_names = load_lfw_dataset(
        min_faces=processor.dp_config.lfw_min_faces_per_person,
        resize_factor=processor.dp_config.lfw_resize_factor
    )

    images = images[:20]
    labels = labels[:20]

    if not len(images):
        print("Nenhuma imagem para processar. Encerrando o script.")
        return

    
    print("\n--- FASE 1: PRÉ-PROCESSANDO IMAGENS (ALINHAMENTO E QUALIDADE) ---")
    processed_faces = []
    corresponding_labels = []

    for i in tqdm(range(len(images)), desc="Pré-processando faces"):
        img = images[i]
        label = labels[i]
        
        aligned_face = processor.preprocess_image(img)
        
        if aligned_face is not None:
            processed_faces.append(aligned_face)
            corresponding_labels.append(target_names[label])

    print(f"  -> {len(processed_faces)} de {len(images)} imagens passaram nos filtros de qualidade.")

    if not processed_faces:
        print("Nenhuma face passou no pré-processamento. Encerrando o script.")
        return

    # --- FASE 2: EXTRAÇÃO DOS EMBEDDINGS ---
    print("\n--- FASE 2: EXTRAINDO EMBEDDINGS DAS FACES PROCESSADAS ---")
    final_embeddings = []
    final_labels = []

    for i in tqdm(range(len(processed_faces)), desc="Extraindo embeddings"):
        face = processed_faces[i]
        label = corresponding_labels[i]

        embedding = processor.extract_embedding(face)
        
        if embedding is not None:
            final_embeddings.append(embedding)
            final_labels.append(label)

    print(f"  -> {len(final_embeddings)} embeddings extraídos com sucesso.")

    # --- FASE 3: SALVANDO OS RESULTADOS ---
    print("\n--- FASE 3: SALVANDO DADOS PROCESSADOS ---")
    
    # Converte a lista de embeddings para um array numpy 2D
    final_embeddings = np.array(final_embeddings)

    output_data = {
        'embeddings': final_embeddings,
        'nomes': final_labels
    }
    
    return output_data