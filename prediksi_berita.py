import joblib
from joblib import load
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

rf_model = load('./saved_model/rf_model.joblib')
svm_model = load('./saved_model/svm_model.joblib')
Tfidf_vect = load('./saved_model/Tfidf_vect.joblib')

st.title('Prediksi Berita Hoax')

artikel_berita = st.text_input('Masukkan artikel berita')

predict = ''

factory = StemmerFactory()
stemmer = factory.create_stemmer()
# Tfidf_vect = TfidfVectorizer()

def preprocess_input(input_text):
  lower = stemmer.stem(input_text.lower())
  tokens = word_tokenize(lower)
  return tokens

def transform_input(input_tokens):
  input_tfidf = Tfidf_vect.transform([" ".join(input_tokens)])
  return input_tfidf

def predict_svm(input_tfidf):
  return svm_model.predict(input_tfidf)

def predict_rf(input_tfidf):
  return rf_model.predict(input_tfidf)

def outputLabel(prediction):
  if prediction == 0:
    return "Berita Hoaks"
  elif prediction == 1:
    return "Bukan Berita Hoaks"

def manualTesting(artikel):
  preprocessed_input = preprocess_input(artikel)
  transformed_input = transform_input(preprocessed_input)
  predicted_svm = predict_svm(transformed_input[0])
  predicted_rf = predict_rf(transformed_input[0])
  return ("Prediksi dengan model SVM adalah " + outputLabel(predicted_svm), "Prediksi dengan model Random Forest adalah " + outputLabel(predicted_rf))

if st.button('Mulai Prediksi'):
  test_artikel = ' - Wasabi adalah pasta hijau pedas yang sering disajikan dengan masakan Jepang, terutama sushi dan sashimi, untuk menambah rasa pada sausnya.Banyak yang menganggap wasabi sebagai makanan super karena kaya akan vitamin C dan juga memiliki sejumlah sifat antibakteri.Namun, mengapa wasabi sangat pedas? Dan apa saja bahan untuk membuat wasabi?Mungkin masih banyak dari kita yang bertanya-tanya, sebenarnya, wasabi terbuat dari apa?Jawaban sebenarnya adalah hanya ada satu bahan wasabi, yaitu wasabi. Untuk membuat wasabi, cukup dengan memarut rimpang tanaman Wasabia Japonica, yang langsung menjadi pasta wasabi yang siap dikonsumsi. Tidak ada campuran atau bahan lain yang digunakan.Namun, sebagian besar bumbu yang disajikan sebagai wasabi di restoran sebenarnya bukanlah wasabi asli.Wasabi palsu ini sebenarnya adalah lobak yang biasanya dicampur dengan mustard dan pewarna makanan hijau. Inilah sebabnya akar putih ini tampak hijau dan mungkin juga mengapa wasabi disalahartikan sebagai lobak pedas.Bahan pengental seperti tepung maizena serta bahan penstabil kimia juga umum digunakan untuk membuat wasabi palsu.Wasabi asli memiliki kepedasan dan rasa yang terkonsentrasi di dalam batang rimpangnya.Wasabi baru terasa pedas setelah batangnya diparut untuk menjadi pasta dan melepaskan rasa serta kepedasannya.Pemarutan memungkinkan fitokimia yang bertanggung jawab atas kepedasan wasabi bereaksi dengan udara karena sebagian besar tersebar di udara.Terkadang,brasa pedas wasabi asli terasa menusuk hidung saat diparut karena sifatnya yang mengudara.'
  st.write(manualTesting(test_artikel)[0])
  st.write(manualTesting(test_artikel)[1])