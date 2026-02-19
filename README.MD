# Proyek Analisis Data: E-Commerce Public Dataset

Proyek ini adalah tugas akhir dari kelas Belajar Fundamental Analisis Data. Fokus utamanya menganalisis data transaksi e-commerce publik dari Brazil dan membangun dashboard sederhana dengan Streamlit.

## Pertanyaan Bisnis
1. Bagaimana tren jumlah pesanan dan total pendapatan perusahaan dari waktu ke waktu?
2. Kategori produk apa yang memiliki angka penjualan tertinggi dan terendah?
3. Bagaimana distribusi tingkat kepuasan pelanggan berdasarkan skor ulasan?
4. Di wilayah mana letak demografi pelanggan terbanyak kita?
5. Bagaimana segmentasi pelanggan berdasarkan perilaku belanja mereka (RFM Analysis)?

## Struktur Direktori
* /Dashboard: Berisi file dashboard.py untuk menjalankan Streamlit dan dataset ecommerce_cleaned.parquet.
* /Data: Kumpulan dataset mentah CSV.
* main_v2.ipynb: Jupyter Notebook berisi alur analisis dari pembersihan data hingga kesimpulan.
* requirements.txt: Daftar pustaka Python yang dibutuhkan.
* url.txt: Tautan menuju dashboard Streamlit.

## Cara Menjalankan Proyek Secara Lokal

### Persiapan
1. Buka terminal dan arahkan ke direktori proyek.
2. Instal pustaka dengan perintah:
   pip install -r requirements.txt

### Menjalankan Jupyter Notebook
1. Pada terminal, ketik perintah:
   jupyter notebook
2. Buka file main_v2.ipynb melalui peramban untuk melihat proses analisis data.

### Menjalankan Dashboard Streamlit
1. Pada terminal, ketik perintah:
   streamlit run dashboard/dashboard.py
2. Buka tautan http://localhost:8501 di peramban.

## Hasil Analisis Utama
* Puncak penjualan terjadi pada bulan November.
* Kategori bed_bath_table dan health_beauty menghasilkan pendapatan tertinggi.
* Mayoritas pelanggan memberi ulasan bintang lima.
* Konsentrasi pelanggan terbanyak berada di wilayah tenggara Brazil, khususnya São Paulo dan Rio de Janeiro.
* Segmen pelanggan didominasi oleh Potential Customers dan Loyal Customers, tetapi ada jumlah Lost Customers yang perlu ditarik kembali lewat strategi CRM.