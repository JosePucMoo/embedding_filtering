import re
import numpy as np

def clean_text(text):
    text = remove_punctuation_marks(text)
    words = extract_unique_words(text)
    return words

def remove_punctuation_marks(text: str) -> str:
    return re.sub(r"[.,:;!?¿¡()\[\]{}\"'«»\-]", '', text.lower())

def extract_unique_words(text: str) -> str:
    return set(text.split())

def load_embeddings(file_vec):
    embeddings = {}
    with open(file_vec, 'r', encoding='utf-8') as f:
        # La primera línea contiene el número de palabras y dimensiones
        _, dimensions = f.readline().split()
        dimensions = int(dimensions)

        # Cargar los vectores de las palabras
        for line in f:
            data = line.strip().split()
            word = data[0]
            vector = np.array(data[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

with open("el_alquimista.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocabulary = clean_text(text)
print(f"Vocabulario extraído: {len(vocabulary)} palabras.")

# Cargar los embeddings desde el archivo .vec
file_vec = "cc.es.300.vec"  # Nombre del archivo .vec que ya tienes
embeddings = load_embeddings(file_vec)

# Paso 3: Filtrar los embeddings solo para las palabras del vocabulario
embeddings_filtered = {}
for word in vocabulary:
    if word in embeddings:
        embeddings_filtered[word] = embeddings[word]

print(f"Embeddings extraídos: {len(embeddings_filtered)} palabras con vectores.")

# Paso 4: Guardar los embeddings en un archivo de texto
with open("embeddings_el_alquimista.txt", "w", encoding="utf-8") as f:
    for word, vector in embeddings_filtered.items():
        vector_str = " ".join(map(str, vector))
        f.write(f"{word} {vector_str}\n")

print("Archivo de embeddings guardado como 'embeddings_el_alquimista.txt'.")
