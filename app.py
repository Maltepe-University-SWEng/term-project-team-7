from flask import Flask, render_template, request
import requests

app = Flask(__name__)


def generate_joke_from_ollama(category):
    """
    Ollama'dan belirtilen kategoride fıkra üretir
    """
    url = "http://localhost:11434/api/generate"

    # Kategori bazında prompt oluşturma
    if category == 'karadeniz':
        prompt = "Bir Karadeniz/Temel fıkrası yaz. Fıkra Türkçe olmalı ve 5-10 cümle içermeli."
    elif category == 'nasreddin_hoca':
        prompt = "Bir Nasreddin Hoca fıkrası yaz. Fıkra Türkçe olmalı ve 5-10 cümle içermeli."
    elif category == 'anadolu':
        prompt = "Bir Anadolu fıkrası yaz. Fıkra geleneksel Türk kültüründen olmalı ve 5-10 cümle içermeli."
    else:
        return "Geçersiz kategori seçildi."

    data = {
        "model": "deepseek-coder",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'Fıkra üretilemedi.')
        else:
            return f"API hatası: {response.status_code}"
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


@app.route('/', methods=['GET', 'POST'])
def index():
    joke = None
    category_name = None

    if request.method == 'POST':
        category = request.form.get('category')

        # Kategori adını Türkçe olarak ayarlama
        if category == 'karadeniz':
            category_name = "Karadeniz"
        elif category == 'nasreddin_hoca':
            category_name = "Nasreddin Hoca"
        elif category == 'anadolu':
            category_name = "Anadolu"

        # Deepseek Coder modelinden fıkra üretme
        joke = generate_joke_from_ollama(category)

    return render_template('index.html', joke=joke, category=category_name)


if __name__ == '__main__':
    app.run(debug=True)