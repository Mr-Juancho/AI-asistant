# download_nltk.py
import nltk

print("Descargando paquetes de NLTK necesarios (puede tardar un momento)...")

# Paquetes que 'unstructured' suele necesitar
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

print("¡Descarga de NLTK completada!")