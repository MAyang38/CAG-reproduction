from sentence_transformers import SentenceTransformer
import json

# load pre-trained model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# read CRSB-Texts.json file   please change your data path
with open('E:\\study\\ECNU\\biye\\RAG\\CAG\\Dataset\\CRSB-Texts.json', 'r') as f:
    data = json.load(f)

# Used to store the converted embedded vector data
new_data = {}
for category, values in data.items():
    new_values = {}
    for key, text_list in values.items():
        if isinstance(text_list, list) and text_list:
            embeddings = model.encode(text_list)
            new_values[key] = embeddings.tolist()
        else:
            new_values[key] = text_list
    new_data[category] = new_values

# Save the converted data as CRSB-Embeddings-MPNET.json
with open('E:\\study\\ECNU\\biye\\RAG\\CAG\\Dataset\\CRSB-Embeddings-MPNET.json', 'w') as f:
    json.dump(new_data, f)