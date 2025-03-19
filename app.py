from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    joke = None
    category = None

    if request.method == 'POST':
        category = request.form.get('category')
        # Şimdilik basit bir fıkra döndürelim
        if category == 'karadeniz':
            joke = "Temel fıkrası buraya gelecek."
        elif category == 'nasreddin_hoca':
            joke = "Nasreddin Hoca fıkrası buraya gelecek."
        elif category == 'anadolu':
            joke = "Anadolu fıkrası buraya gelecek."

    return render_template('index.html', joke=joke, category=category)


if __name__ == '__main__':
    app.run(debug=True)