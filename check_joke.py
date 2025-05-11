import json
import argparse
import os
from difflib import SequenceMatcher
import re


def load_jokes():
    jokes = []

    try:
        # Tüm olası konumları kontrol et
        file_paths = [
            'fikralar.json',  # kök dizinde
            os.path.join('data', 'fikralar.json'),  # data klasöründe
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fikralar.json')  # betik ile aynı dizinde
        ]

        for path in file_paths:
            if os.path.exists(path):
                print(f"fikralar.json bulundu: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    try:
                        fikralar_data = json.load(f)
                        for item in fikralar_data:
                            if "metin" in item and item["metin"].strip():
                                jokes.append({
                                    "type": item.get("kategori", "Genel").lower(),
                                    "text": item["metin"].strip(),
                                    "title": item.get("baslik", "")
                                })
                        print(f"{path} dosyasından {len(fikralar_data)} fıkra yüklendi.")
                        break  # Başarıyla yükledikten sonra döngüyü kır
                    except json.JSONDecodeError as e:
                        print(f"JSON ayrıştırma hatası ({path}): {e}")

        # Eski dosyaları da yüklemeye devam et
        nasreddin_loaded = False
        temel_loaded = False

        # Nasreddin fıkraları
        for path in ['nasreddin.json', os.path.join('data', 'nasreddin.json')]:
            if os.path.exists(path) and not nasreddin_loaded:
                with open(path, 'r', encoding='utf-8') as f:
                    try:
                        nasreddin_data = json.load(f)
                        count = 0
                        for item in nasreddin_data:
                            if "dc_Fikra" in item and item["dc_Fikra"].strip():
                                jokes.append({"type": "nasreddin", "text": item["dc_Fikra"].strip()})
                                count += 1
                        print(f"{path} dosyasından {count} Nasreddin fıkrası yüklendi.")
                        nasreddin_loaded = True
                    except json.JSONDecodeError as e:
                        print(f"JSON ayrıştırma hatası ({path}): {e}")

        # Temel fıkraları
        for path in ['temel.json', os.path.join('data', 'temel.json')]:
            if os.path.exists(path) and not temel_loaded:
                with open(path, 'r', encoding='utf-8') as f:
                    try:
                        temel_data = json.load(f)
                        count = 0
                        for item in temel_data:
                            if "dc_Fikra" in item and item["dc_Fikra"].strip():
                                jokes.append({"type": "temel", "text": item["dc_Fikra"].strip()})
                                count += 1
                        print(f"{path} dosyasından {count} Temel fıkrası yüklendi.")
                        temel_loaded = True
                    except json.JSONDecodeError as e:
                        print(f"JSON ayrıştırma hatası ({path}): {e}")

        print(f"Toplam {len(jokes)} fıkra yüklendi.")
        return jokes

    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return []


def clean_text(text):
    # Remove punctuation and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def check_joke(joke_text, jokes, threshold=0.8):
    if not jokes:
        return False, 0, None

    # Clean the input text
    cleaned_query = clean_text(joke_text)

    # Track best match
    max_similarity = 0
    best_match = None

    for joke in jokes:
        original_text = joke["text"]
        cleaned_original = clean_text(original_text)

        # Calculate similarity
        similarity = SequenceMatcher(None, cleaned_query, cleaned_original).ratio()

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = joke

    # Determine if it's a match
    if max_similarity >= threshold:
        return True, max_similarity, best_match
    elif max_similarity >= 0.5:
        return "maybe", max_similarity, best_match
    else:
        return False, max_similarity, best_match


def display_results(is_match, similarity, match, joke_text):
    similarity_percent = round(similarity * 100, 2)

    print("\n======= FIKRA KONTROL SONUÇLARI =======")
    print(f"Kontrol edilen fıkra: \"{joke_text[:50]}...\"")
    print(f"Benzerlik oranı: {similarity_percent}%")

    if is_match is True:
        print("\n✅ BU FIKRA VERİ SETİNDE BULUNUYOR!")
        print(f"Tür: {match['type'].capitalize()} Fıkrası")
        if 'title' in match and match['title']:
            print(f"Başlık: {match['title']}")
        print("\nOrijinal fıkra:")
        print(f"\"{match['text']}\"")
    elif is_match == "maybe":
        print("\n⚠️ BU FIKRA VERİ SETİNDEKİ BİR FIKRAYA BENZER OLABİLİR")
        print(f"Tür: {match['type'].capitalize()} Fıkrası")
        if 'title' in match and match['title']:
            print(f"Başlık: {match['title']}")
        print("\nEn yakın eşleşme:")
        print(f"\"{match['text']}\"")
    else:
        print("\n❌ BU FIKRA VERİ SETİNDE BULUNMUYOR")
        print("Bu muhtemelen model tarafından üretilmiş yeni bir fıkradır.")
        if match:
            print("\nEn yakın fıkra (düşük benzerlik):")
            print(f"\"{match['text']}\"")


def main():
    parser = argparse.ArgumentParser(description='Fıkra veri seti kontrol aracı')
    parser.add_argument('--file', '-f', help='Kontrol edilecek fıkranın bulunduğu metin dosyası')
    parser.add_argument('--text', '-t', help='Doğrudan kontrol edilecek fıkra metni')
    parser.add_argument('--threshold', '-th', type=float, default=0.6,
                        help='Eşleşme için benzerlik eşiği (0-1 arası, varsayılan: 0.6)')

    args = parser.parse_args()

    # Load jokes
    all_jokes = load_jokes()

    if not all_jokes:
        print("Veri seti yüklenemedi. Lütfen veri dosyalarını kontrol edin.")
        return

    # Get joke text from file or command line
    joke_text = ""
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                joke_text = f.read().strip()
        except Exception as e:
            print(f"Dosya okuma hatası: {e}")
            return
    elif args.text:
        joke_text = args.text
    else:
        print("Lütfen bir fıkra metni veya dosya girin.")
        joke_text = input("Kontrol edilecek fıkrayı girin: ").strip()

    if not joke_text:
        print("Boş fıkra metni kontrol edilemez.")
        return

    # Check the joke
    is_match, similarity, match = check_joke(joke_text, all_jokes, args.threshold)

    # Display results
    display_results(is_match, similarity, match, joke_text)


if __name__ == "__main__":
    main()