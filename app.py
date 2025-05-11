from flask import Flask, render_template, request, jsonify
import json
import random
import torch
import os
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Global değişkenler - modelleri bir kere yükle
global_gpt2_model = None
global_gpt2_tokenizer = None
global_lstm_model = None
global_lstm_char_to_idx = None
global_lstm_idx_to_char = None
all_jokes = []


# LSTM model sınıfı tanımı
class JokeLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=3, dropout=0.5):
        super(JokeLSTM, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        return (h0, c0)


# Fıkra kalitesini değerlendiren fonksiyon
def score_joke(joke_text):
    """Fıkranın kalitesini değerlendiren skorlama fonksiyonu"""
    score = 0

    # Uzunluk puanı - daha uzun fıkralar genellikle daha iyidir (belli bir limite kadar)
    words = joke_text.split()
    if len(words) < 10:
        score -= 10  # Çok kısa fıkralar kötüdür
    elif len(words) > 60:
        score += 15  # Uzun fıkralar iyidir
    else:
        score += len(words) / 4  # Orta uzunluktaki fıkralar için orantılı puan

    # Dilbilgisi puanı - cümle sonu noktalama işaretleri
    if joke_text.endswith(('.', '!', '?')):
        score += 5  # Düzgün biten fıkralar daha iyi

    # Tutarlılık puanı - başında "iki" kelimesi var mı (model sorunuydu)
    if joke_text.lower().startswith("iki"):
        score -= 5  # "iki" ile başlayan fıkralardan kaçın

    # Dialog puanı - konuşma içeren fıkralar genellikle daha iyidir
    if '-' in joke_text or '"' in joke_text:
        score += 10

    # Karakter puanı - popüler karakterleri içeriyor mu?
    if "Temel" in joke_text or "Nasreddin" in joke_text or "Hoca" in joke_text:
        score += 8

    # Anlam puanı - saçma sapan kelime dizileri içermiyor mu?
    nonsense_patterns = ["nann", "ÇALAR", "ĞANIN", "BURADA K", "GEL"]
    for pattern in nonsense_patterns:
        if pattern in joke_text:
            score -= 8

    return score


# Fıkra metni son işleme
def post_process_joke(text):
    """Üretilen fıkra metnini iyileştirme"""
    # Çok uzun cümleleri kısaltma
    if len(text) > 500:
        text = text[:500]

    # Son cümleyi düzgün bir şekilde tamamlama
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences and not sentences[-1].endswith(('.', '!', '?')):
        # Son cümle düzgün bitmediyse, önceki cümleleri al
        if len(sentences) > 1:
            text = ' '.join(sentences[:-1])
        else:
            # Tek bir cümle varsa, sonuna nokta ekle
            text = sentences[0] + '.'

    # Anlamsız kelimeleri veya karakter dizilerini temizleme
    nonsense_patterns = ["KAĞAŞ", "ÇALAR", "ĞANIN", "ÇAŞAR", "BÇOCUK"]
    for pattern in nonsense_patterns:
        text = text.replace(pattern, "")

    # İlk harf büyük yapma
    if text and len(text) > 0:
        text = text[0].upper() + text[1:]

    return text


# Veri setlerini yükle
def load_joke_datasets():
    global all_jokes
    try:
        # Dosya yollarını kontrol et
        data_dir = "data"
        nasreddin_path = os.path.join(data_dir, "nasreddin.json")
        temel_path = os.path.join(data_dir, "temel.json")
        genel_path = os.path.join(data_dir, "genel.json")  # Genel fıkralar için

        if not os.path.exists(nasreddin_path):
            nasreddin_path = "nasreddin.json"  # Ana dizine düş

        if not os.path.exists(temel_path):
            temel_path = "temel.json"  # Ana dizine düş

        if not os.path.exists(genel_path):
            genel_path = None  # Genel fıkra dosyası yoksa None olarak ayarla

        # Dosyaları yükle
        with open(nasreddin_path, 'r', encoding='utf-8') as f:
            nasreddin_data = json.load(f)

        with open(temel_path, 'r', encoding='utf-8') as f:
            temel_data = json.load(f)

        # Genel fıkralar varsa yükle
        genel_data = []
        if genel_path and os.path.exists(genel_path):
            with open(genel_path, 'r', encoding='utf-8') as f:
                genel_data = json.load(f)

        # Tüm fıkraları birleştir
        all_jokes = []
        for item in nasreddin_data:
            if "dc_Fikra" in item and item["dc_Fikra"].strip():
                all_jokes.append({"type": "nasreddin", "text": item["dc_Fikra"].strip()})

        for item in temel_data:
            if "dc_Fikra" in item and item["dc_Fikra"].strip():
                all_jokes.append({"type": "temel", "text": item["dc_Fikra"].strip()})

        for item in genel_data:
            if "dc_Fikra" in item and item["dc_Fikra"].strip():
                all_jokes.append({"type": "genel", "text": item["dc_Fikra"].strip()})

        print(f"Toplam {len(all_jokes)} fıkra yüklendi.")
        return True
    except Exception as e:
        print(f"Veri setleri yüklenirken hata oluştu: {e}")
        all_jokes = []
        return False


# GPT-2 modelini yükle
def load_gpt2_model():
    global global_gpt2_model, global_gpt2_tokenizer

    print("GPT-2 modeli yükleniyor...")
    model_path = "./models/fikra_model"  # Modelin yolu
    try:
        # Cihazı belirle
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Cihaz: {device}")

        # Tokenizer'ı yükle
        global_gpt2_tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Pad token ve attention mask için gerekli ayarlar
        if global_gpt2_tokenizer.pad_token is None:
            global_gpt2_tokenizer.pad_token = global_gpt2_tokenizer.eos_token
            global_gpt2_tokenizer.pad_token_id = global_gpt2_tokenizer.eos_token_id

        # Modeli yükle
        global_gpt2_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            pad_token_id=global_gpt2_tokenizer.eos_token_id
        )

        # Modeli değerlendirme moduna al
        global_gpt2_model.to(device)
        global_gpt2_model.eval()

        print("GPT-2 modeli yüklendi!")
        return True
    except Exception as e:
        print(f"GPT-2 modeli yüklenirken hata oluştu: {e}")
        # Hata durumunda model ve tokenizer'ı None olarak ayarla
        global_gpt2_model = None
        global_gpt2_tokenizer = None
        return False


# LSTM modelini yükle
def load_lstm_model():
    global global_lstm_model, global_lstm_char_to_idx, global_lstm_idx_to_char

    print("LSTM modeli yükleniyor...")
    model_path = "./models/lstm_model/best_model.pt"
    try:
        # Cihazı belirle
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Cihaz: {device}")

        # Checkpoint'i yükle
        checkpoint = torch.load(model_path, map_location=device)

        # Model parametrelerini al
        vocab_size = checkpoint['vocab_size']
        embedding_dim = 256
        hidden_dim = 512
        num_layers = 3
        dropout = 0.5

        # Karakter dizinlerini al
        global_lstm_char_to_idx = checkpoint['char_to_idx']
        global_lstm_idx_to_char = checkpoint['idx_to_char']

        # Modeli oluştur ve ağırlıkları yükle
        global_lstm_model = JokeLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)
        global_lstm_model.load_state_dict(checkpoint['model_state_dict'])
        global_lstm_model.eval()

        print("LSTM modeli yüklendi!")
        return True
    except Exception as e:
        print(f"LSTM modeli yüklenirken hata oluştu: {e}")
        # Hata durumunda model ve tokenizer'ı None olarak ayarla
        global_lstm_model = None
        global_lstm_char_to_idx = None
        global_lstm_idx_to_char = None
        return False


# Uygulama başlatıldığında veri setlerini ve modelleri yükle
@app.before_first_request
def initialize():
    load_joke_datasets()
    load_gpt2_model()
    load_lstm_model()


# GPT-2 modeli ile fıkra üretme
def generate_gpt2_joke(joke_type="random", max_attempts=3):
    global global_gpt2_model, global_gpt2_tokenizer

    # Model yüklü değilse hata döndür
    if global_gpt2_model is None or global_gpt2_tokenizer is None:
        print("GPT-2 modeli yüklenemedi")
        return None, None, False

    # Kategori belirle
    if joke_type == "nasreddin":
        kategori = "Nasreddin"
        joke_type_display = "Nasreddin Hoca Fıkrası"
    elif joke_type == "temel":
        kategori = "Temel"
        joke_type_display = "Temel Fıkrası"
    elif joke_type == "genel":
        kategori = "Genel"
        joke_type_display = "Genel Fıkra"
    else:
        # Rastgele seçim
        kategori = random.choice(["Nasreddin", "Temel", "Genel"])
        joke_type_display = "Nasreddin Hoca Fıkrası" if kategori == "Nasreddin" else "Temel Fıkrası" if kategori == "Temel" else "Genel Fıkra"

    # Farklı prompt formatları
    prompt_templates = [
        f"<{kategori}> Bir {kategori} fıkrası: ",
        f"<{kategori}> Komik bir {kategori} fıkrası: ",
        f"<{kategori}> Şöyle bir {kategori} fıkrası anlatılır: "
    ]

    # Farklı üretim parametreleri
    param_sets = [
        # Tutarlı parametreler
        {"temperature": 0.7, "top_p": 0.92, "top_k": 50, "repetition_penalty": 1.2, "max_length": 200},
        # Yaratıcı parametreler
        {"temperature": 0.8, "top_p": 0.95, "top_k": 60, "repetition_penalty": 1.1, "max_length": 220},
        # Dengeli parametreler
        {"temperature": 0.75, "top_p": 0.9, "top_k": 55, "repetition_penalty": 1.15, "max_length": 210}
    ]

    best_joke = None
    best_score = -float('inf')
    is_from_dataset = False

    for attempt in range(max_attempts):
        try:
            # Her denemede farklı prompt ve parametre kombinasyonu
            prompt = prompt_templates[attempt % len(prompt_templates)]
            params = param_sets[attempt % len(param_sets)]

            # Girdiyi tokenize et ve attention mask oluştur
            inputs = global_gpt2_tokenizer(prompt, return_tensors="pt", padding=True)

            # Girdiyi doğru cihaza taşı
            device = next(global_gpt2_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Çıktı üret
            with torch.no_grad():
                outputs = global_gpt2_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=params["max_length"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=params["top_k"],
                    repetition_penalty=params["repetition_penalty"],
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=global_gpt2_tokenizer.eos_token_id
                )

            # Tokenleri decode et
            generated_text = global_gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Prompt kısmını temizleme
            joke_text = generated_text.replace(f"<{kategori}>", "")
            patterns = [
                f"Bir {kategori} fıkrası:",
                f"Komik bir {kategori} fıkrası:",
                f"Şöyle bir {kategori} fıkrası anlatılır:"
            ]
            for pattern in patterns:
                joke_text = joke_text.replace(pattern, "")
            joke_text = joke_text.strip()

            # Fıkrayı son işleme
            joke_text = post_process_joke(joke_text)

            # Fıkra kalitesini değerlendir
            score = score_joke(joke_text)

            # En iyi fıkrayı güncelle
            if score > best_score:
                best_joke = joke_text
                best_score = score

            # Fıkranın veri setinde olup olmadığını kontrol et
            actual_joke_type = "nasreddin" if kategori == "Nasreddin" else "temel" if kategori == "Temel" else "genel"
            is_from_dataset = is_joke_in_dataset(joke_text, actual_joke_type)

            # Yeterince iyi bir fıkra üretildiyse döngüden çık
            if score > 15 and not is_from_dataset:
                break

        except Exception as e:
            print(f"GPT-2 model hatası (Deneme {attempt + 1}): {e}")
            if attempt == max_attempts - 1:
                # Son deneme başarısız oldu
                break

    # Hiç fıkra üretilemediyse
    if best_joke is None:
        return None, None, False

    return best_joke, joke_type_display, is_from_dataset


# LSTM modeli ile fıkra üretme
def generate_lstm_joke(joke_type="random", temperature=0.7):
    global global_lstm_model, global_lstm_char_to_idx, global_lstm_idx_to_char

    # Model yüklü değilse hata döndür
    if global_lstm_model is None or global_lstm_char_to_idx is None or global_lstm_idx_to_char is None:
        print("LSTM modeli yüklenemedi")
        return None, None, False

    # Kategori belirle
    if joke_type == "nasreddin":
        category = "Nasreddin"
        joke_type_display = "Nasreddin Hoca Fıkrası"
    elif joke_type == "temel":
        category = "Temel"
        joke_type_display = "Temel Fıkrası"
    elif joke_type == "genel":
        category = "Genel"
        joke_type_display = "Genel Fıkra"
    else:
        # Rastgele seçim
        category = random.choice(["Temel", "Nasreddin", "Genel"])
        joke_type_display = "Nasreddin Hoca Fıkrası" if category == "Nasreddin" else "Temel Fıkrası" if category == "Temel" else "Genel Fıkra"

    max_attempts = 3
    best_joke = None
    best_score = -float('inf')
    is_from_dataset = False

    for attempt in range(max_attempts):
        try:
            # Başlangıç metni
            seed_text = f"<BASLA><{category}>"

            # İndeks çevirisi
            chars = [global_lstm_char_to_idx.get(ch, 0) for ch in seed_text]
            generated = seed_text

            # Gizli durumu sıfırla
            device = next(global_lstm_model.parameters()).device
            hidden = global_lstm_model.init_hidden(1, device)

            # Karakter karakter üretme - daha uzun maximum uzunluk
            max_length = 500  # Daha uzun fıkralar için
            with torch.no_grad():
                for _ in range(max_length):
                    # Son karakteri al
                    x = torch.tensor([[chars[-1]]]).to(device)

                    # Tahmin yap
                    output, hidden = global_lstm_model(x, hidden)
                    output = output.squeeze().div(temperature).exp()

                    # Multinomial sampling
                    top_char = torch.multinomial(output, 1)[0]

                    # Karakteri ekle
                    next_char = global_lstm_idx_to_char.get(top_char.item(), ' ')
                    generated += next_char
                    chars.append(top_char.item())

                    # Bitiş etiketi kontrolü
                    if "<BITIR>" in generated:
                        break

            # Bitiş etiketine kadar olan metni al
            if "<BITIR>" in generated:
                generated = generated.split("<BITIR>")[0]

            # Başlangıç etiketlerini temizle
            joke_text = generated.replace("<BASLA>", "").replace(f"<{category}>", "").strip()

            # Kalite kontrolü - daha az kısıtlayıcı
            if len(joke_text.split()) < 5:
                print("LSTM fıkra kalitesi çok düşük, tekrar deneniyor...")
                continue

            # Fıkra kalitesini değerlendir
            score = score_joke(joke_text)

            # En iyi fıkrayı güncelle
            if score > best_score:
                best_joke = joke_text
                best_score = score

            # Fıkranın veri setinde olup olmadığını kontrol et
            joke_type_for_dataset = "nasreddin" if category == "Nasreddin" else "temel" if category == "Temel" else "genel"
            is_from_dataset = is_joke_in_dataset(joke_text, joke_type_for_dataset)

            # Yeterince iyi bir fıkra üretildiyse
            if score > 10:
                break

        except Exception as e:
            print(f"LSTM model hatası (Deneme {attempt + 1}): {e}")
            if attempt == max_attempts - 1:
                # Son deneme başarısız oldu
                break

    # Hiç fıkra üretilemediyse
    if best_joke is None:
        return None, None, False

    return best_joke, joke_type_display, is_from_dataset


# Fıkranın veri setinde olup olmadığını kontrol et
def is_joke_in_dataset(joke_text, joke_type):
    global all_jokes

    # Kısa metinlerde ilk birkaç kelimeyi kontrol etmek yeterli olabilir
    joke_summary = ' '.join(joke_text.split()[:10]).lower()

    for joke in all_jokes:
        if joke["type"] == joke_type:
            original_text = joke["text"].lower()

            # İlk kontrol: ilk 50 karakter
            if joke_text[:50].lower() in original_text or original_text[:50] in joke_text.lower():
                return True

            # İkinci kontrol: ilk birkaç kelime
            original_summary = ' '.join(original_text.split()[:10])
            if joke_summary in original_summary or original_summary in joke_summary:
                return True

            # Üçüncü kontrol: Jaccard benzerliği
            joke_words = set(joke_text.lower().split())
            original_words = set(original_text.split())

            # Kesişim / Birleşim
            if len(joke_words) > 0 and len(original_words) > 0:
                jaccard = len(joke_words.intersection(original_words)) / len(joke_words.union(original_words))
                if jaccard > 0.7:  # Yüksek benzerlik
                    return True

    return False


# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')


# Fıkra üretme API'si
@app.route('/generate_joke', methods=['POST'])
def generate_joke_api():
    # Başlangıç zamanını kaydet
    start_time = time.time()

    # İstek parametrelerini al
    joke_type = request.form.get('type', 'random')
    model_type = request.form.get('model', 'gpt2')  # Model türü

    # Modele göre fıkra üret
    if model_type == "lstm":
        joke_text, joke_type_display, is_from_dataset = generate_lstm_joke(joke_type)
        model_display = "LSTM Modeli"
    else:  # Varsayılan olarak GPT-2
        joke_text, joke_type_display, is_from_dataset = generate_gpt2_joke(joke_type)
        model_display = "GPT-2 Modeli"

    # Geçen süreyi hesapla
    processing_time = time.time() - start_time
    print(f"Fıkra üretim süresi: {processing_time:.2f} saniye")

    # Fıkra üretilemediyse hata mesajı döndür
    if joke_text is None:
        return jsonify({
            'success': False,
            'message': 'Fıkra üretilemedi. Lütfen tekrar deneyin.',
            'type': joke_type_display or (
                'Nasreddin Hoca Fıkrası' if joke_type == 'nasreddin' else 'Temel Fıkrası' if joke_type == 'temel' else 'Genel Fıkra'),
            'model': model_display
        })

    # Başarılı yanıt oluştur
    return jsonify({
        'success': True,
        'joke': joke_text,
        'type': joke_type_display,
        'model': model_display,
        'from_dataset': is_from_dataset,
        'processing_time': f"{processing_time:.2f}"
    })


# Karşılaştırma API'si
@app.route('/compare_models', methods=['POST'])
def compare_models_api():
    # İstek parametrelerini al
    joke_type = request.form.get('type', 'random')

    # Her iki modelden de fıkra üret
    gpt2_joke, gpt2_type, gpt2_from_dataset = generate_gpt2_joke(joke_type)
    lstm_joke, lstm_type, lstm_from_dataset = generate_lstm_joke(joke_type)

    # Sonuçları döndür
    return jsonify({
        'gpt2': {
            'success': gpt2_joke is not None,
            'joke': gpt2_joke or 'Fıkra üretilemedi. Lütfen tekrar deneyin.',
            'type': gpt2_type or (
                'Nasreddin Hoca Fıkrası' if joke_type == 'nasreddin' else 'Temel Fıkrası' if joke_type == 'temel' else 'Genel Fıkra'),
            'from_dataset': gpt2_from_dataset,
            'model': 'GPT-2 Modeli'
        },
        'lstm': {
            'success': lstm_joke is not None,
            'joke': lstm_joke or 'Fıkra üretilemedi. Lütfen tekrar deneyin.',
            'type': lstm_type or (
                'Nasreddin Hoca Fıkrası' if joke_type == 'nasreddin' else 'Temel Fıkrası' if joke_type == 'temel' else 'Genel Fıkra'),
            'from_dataset': lstm_from_dataset,
            'model': 'LSTM Modeli'
        }
    })


# Hakkında sayfası
@app.route('/about')
def about():
    return render_template('about.html')


# İletişim sayfası
@app.route('/contact')
def contact():
    return render_template('contact.html')


# Uygulama başlangıcında veri setlerini yükle
if __name__ == '__main__':
    # Veri setlerini ve modelleri uygulama başlangıcında yükle
    load_joke_datasets()
    load_gpt2_model()
    load_lstm_model()

    # Flask uygulamasını başlat
    app.run(debug=True)









