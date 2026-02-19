import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime as dt

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet("ecommerce_cleaned.parquet")
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # --- TAMBAHKAN KODE INI ---
    # Ubah semua kolom float16 menjadi float32 agar Streamlit tidak error
    kolom_float16 = df.select_dtypes(include=['float16']).columns
    for col in kolom_float16:
        df[col] = df[col].astype('float32')
        
    return df

df_clean = load_data()

st.sidebar.header("Filter Data")
min_date = df_clean['order_purchase_timestamp'].min().date()
max_date = df_clean['order_purchase_timestamp'].max().date()

start_date, end_date = st.sidebar.date_input(
    label='Rentang Waktu',
    min_value=min_date,
    max_value=max_date,
    value=[min_date, max_date]
)

main_df = df_clean[(df_clean['order_purchase_timestamp'].dt.date >= start_date) & 
                   (df_clean['order_purchase_timestamp'].dt.date <= end_date)]

df_orders_unique = main_df.drop_duplicates(subset=['order_id'])
df_items_unique = main_df.drop_duplicates(subset=['order_id', 'price', 'product_category_name'])

st.title("E-Commerce Data Analytics Dashboard")
st.markdown("Menampilkan analisis performa bisnis, demografi pelanggan, dan kualitas layanan dari E-Commerce.")

col1, col2 = st.columns(2)
with col1:
    total_pesanan = df_orders_unique.shape[0]
    st.metric("Total Pesanan (Orders)", value=f"{total_pesanan:,}")
with col2:
    total_pendapatan = df_items_unique['price'].sum()
    st.metric("Total Pendapatan (Revenue)", value=f"${total_pendapatan:,.2f}")

st.markdown("---")

# TREN PENJUALAN
st.subheader("Tren Jumlah Pesanan dan Total Pendapatan Bulanan")
df_items_unique_copy = df_items_unique.copy()
df_items_unique_copy['order_month'] = df_items_unique_copy['order_purchase_timestamp'].dt.to_period('M')
tren_pendapatan = df_items_unique_copy.groupby('order_month')['price'].sum().reset_index()
tren_pendapatan['order_month'] = tren_pendapatan['order_month'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=tren_pendapatan, x='order_month', y='price', marker='o', color="green", ax=ax)
ax.set_ylabel("Total Pendapatan ($)")
ax.set_xlabel("Bulan")
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

# KATEGORI PRODUK PALING LARIS
st.subheader("10 Kategori Produk Paling Laris")
kategori = df_items_unique['product_category_name'].value_counts().head(10).reset_index()
kategori.columns = ['Kategori', 'Jumlah Terjual']

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(data=kategori, x='Jumlah Terjual', y='Kategori', palette='viridis', ax=ax2)
ax2.set_xlabel("Jumlah Terjual")
ax2.set_ylabel("Kategori Produk")
st.pyplot(fig2)

# KEPUASAN PELANGGAN
st.subheader("Distribusi Skor Ulasan Pelanggan")
df_reviews = main_df.dropna(subset=['review_score']).drop_duplicates(subset=['order_id', 'review_score'])
review_counts = df_reviews['review_score'].value_counts().sort_index().reset_index()
review_counts.columns = ['Skor', 'Jumlah']

fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.barplot(data=review_counts, x='Skor', y='Jumlah', palette='coolwarm', ax=ax3)
ax3.set_xlabel("Skor Ulasan (1 - 5)")
ax3.set_ylabel("Jumlah Ulasan")
st.pyplot(fig3)

# PETA PERSEBARAN GEOSPASIAL
st.subheader("Peta Persebaran Demografi Pelanggan (Brazil)")
st.markdown("Peta interaktif persebaran pelanggan. Area dengan kepadatan titik tertinggi menunjukkan konsentrasi demografi pembeli (contoh: São Paulo dan Rio de Janeiro).")
df_geo = main_df.drop_duplicates(subset=['customer_unique_id', 'geolocation_lat', 'geolocation_lng'])

# Filter batas koordinat kotak wilayah negara Brazil
bbox = [
    df_geo['geolocation_lng'].between(-73.9828, -34.7931),
    df_geo['geolocation_lat'].between(-33.7511, 5.2743)
]
df_geo_brazil = df_geo[bbox[0] & bbox[1]].copy()

df_geo_brazil.rename(columns={'geolocation_lat': 'lat', 'geolocation_lng': 'lon'}, inplace=True)

if len(df_geo_brazil) > 20000:
    df_map = df_geo_brazil.sample(n=20000, random_state=42)
else:
    df_map = df_geo_brazil

# Render map menggunakan data sample 
st.map(df_map[['lat', 'lon']], zoom=3)

# RFM
st.subheader("Segmentasi Pelanggan (RFM Analysis)")
tanggal_sekarang = df_items_unique['order_purchase_timestamp'].max() + dt.timedelta(days=1)

rfm = df_items_unique.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (tanggal_sekarang - x.max()).days,
    'order_id': 'nunique',
    'price': 'sum'
}).reset_index()
rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']

rfm['r_rank'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1]).astype(int)
rfm['m_rank'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4]).astype(int)
rfm['f_rank'] = rfm['Frequency'].apply(lambda x: 4 if x > 3 else (3 if x > 2 else (2 if x > 1 else 1)))

rfm['RFM_Total'] = rfm[['r_rank','f_rank','m_rank']].sum(axis=1)

def segment_customer(score):
    if score >= 10: return 'Best Customers'
    elif score >= 7: return 'Loyal Customers'
    elif score >= 5: return 'Potential Customers'
    else: return 'Lost Customers'

rfm['Segment'] = rfm['RFM_Total'].apply(segment_customer)

segmen_counts = rfm['Segment'].value_counts().reset_index()
segmen_counts.columns = ['Segment', 'Total']

fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.barplot(data=segmen_counts, x='Total', y='Segment', palette='Set2', ax=ax4)
ax4.set_xlabel("Jumlah Pelanggan")
ax4.set_ylabel("Segmen")
st.pyplot(fig4)

st.caption("Hak Cipta © 2026 - Proyek Analisis Data E-Commerce oleh Athallah Azhar Aulia Hadi")