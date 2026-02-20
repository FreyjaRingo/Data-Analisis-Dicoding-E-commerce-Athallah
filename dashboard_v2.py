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
    
    # Penyesuaian tipe data float16 menjadi float32 untuk menghindari error pada Streamlit
    kolom_float16 = df.select_dtypes(include=['float16']).columns
    for col in kolom_float16:
        df[col] = df[col].astype('float32')
        
    return df

df_clean = load_data()

# --- SIDEBAR FILTER ---
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
                   (df_clean['order_purchase_timestamp'].dt.date <= end_date)].copy()

st.title("E-Commerce Public Dataset Dashboard")
st.markdown("---")

# --- 1. TREN PENDAPATAN ---
st.subheader("Tren Jumlah Pesanan & Total Pendapatan")
main_df['order_month'] = main_df['order_purchase_timestamp'].dt.to_period('M')
df_trend = main_df.drop_duplicates(subset=['order_id', 'order_item_id']).copy()
tren_pendapatan = df_trend.groupby('order_month')['price'].sum().reset_index()
tren_pendapatan['order_month'] = tren_pendapatan['order_month'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=tren_pendapatan, x='order_month', y='price', marker='o', color="#1f77b4", ax=ax)
ax.set_title("Tren Pendapatan Bulanan", fontsize=14)
ax.set_ylabel("Total Pendapatan ($)")
ax.set_xlabel("Bulan")
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

# --- 2. KATEGORI PRODUK ---
st.subheader("Kategori Produk Paling Laris & Paling Sepi")
kategori_penjualan = main_df['product_category_name_english'].value_counts().reset_index()
kategori_penjualan.columns = ['Kategori', 'Jumlah Terjual']

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

top_categories = kategori_penjualan.head(10)
bottom_categories = kategori_penjualan.tail(10)

# Logika highlight warna
colors_top = ["#1f77b4" if i == 0 else "#D3D3D3" for i in range(len(top_categories))]
sns.barplot(data=top_categories, x='Jumlah Terjual', y='Kategori', palette=colors_top, ax=ax[0])
ax[0].set_title("10 Kategori Produk Terlaris", fontsize=14)
ax[0].set_xlabel("Jumlah Terjual")
ax[0].set_ylabel("Kategori")

colors_bottom = ["#d62728" if i == 0 else "#D3D3D3" for i in range(len(bottom_categories))]
sns.barplot(data=bottom_categories, x='Jumlah Terjual', y='Kategori', palette=colors_bottom, ax=ax[1])
ax[1].set_title("10 Kategori Produk Paling Sepi", fontsize=14)
ax[1].set_xlabel("Jumlah Terjual")
ax[1].set_ylabel("")

plt.tight_layout()
st.pyplot(fig)

# --- 3. SKOR ULASAN ---
st.subheader("Distribusi Skor Ulasan Pelanggan")
df_reviews_unique = main_df.dropna(subset=['review_score']).drop_duplicates(subset=['order_id'])
review_counts = df_reviews_unique['review_score'].value_counts().sort_index().reset_index()
review_counts.columns = ['Skor', 'Jumlah']

max_skor = review_counts['Jumlah'].max()
colors_review = ["#1f77b4" if x == max_skor else "#D3D3D3" for x in review_counts['Jumlah']]

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=review_counts, x='Skor', y='Jumlah', palette=colors_review, ax=ax)
ax.set_title("Distribusi Skor Ulasan Pelanggan", fontsize=14)
ax.set_xlabel("Skor Ulasan (1 - 5)")
ax.set_ylabel("Jumlah Ulasan")
st.pyplot(fig)

# --- 4. GEOSPASIAL ---
st.subheader("Persebaran Demografi Pelanggan (Brazil)")
# Memastikan penamaan kolom sesuai standar Streamlit (lat, lon)
if 'geolocation_lat' in main_df.columns and 'geolocation_lng' in main_df.columns:
    df_map = main_df.dropna(subset=['geolocation_lat', 'geolocation_lng']).copy()
    df_map.rename(columns={'geolocation_lat': 'lat', 'geolocation_lng': 'lon'}, inplace=True)
    
    # Filter batas negara Brazil membuang outlier
    df_map = df_map[(df_map['lon'] >= -73.98) & (df_map['lon'] <= -34.79) & 
                    (df_map['lat'] >= -33.75) & (df_map['lat'] <= 5.27)]
    
    # Batasi sampel render agar dashboard tidak lag
    df_map_sample = df_map[['lat', 'lon']].sample(n=min(10000, len(df_map)), random_state=42)
    st.map(df_map_sample, zoom=3)
else:
    st.warning("Data koordinat tidak ditemukan di dalam dataset. Pastikan Anda telah mengonversi notebook dengan benar.")

# --- 5. RFM ANALYSIS ---
st.subheader("Segmentasi Pelanggan (RFM Analysis)")
df_rfm_base = main_df.drop_duplicates(subset=['order_id', 'order_item_id']).copy()
if not df_rfm_base.empty:
    tanggal_sekarang = df_rfm_base['order_purchase_timestamp'].max() + dt.timedelta(days=1)

    rfm = df_rfm_base.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (tanggal_sekarang - x.max()).days,
        'order_id': 'nunique',
        'price': 'sum'
    }).reset_index()
    rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']

    rfm['r_rank'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop').astype(int)
    rfm['m_rank'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)
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

    max_segmen = segmen_counts['Total'].max()
    colors_rfm = ["#1f77b4" if x == max_segmen else "#D3D3D3" for x in segmen_counts['Total']]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=segmen_counts, x='Total', y='Segment', palette=colors_rfm, ax=ax)
    ax.set_title("Segmentasi Pelanggan Berdasarkan RFM", fontsize=14)
    ax.set_xlabel("Jumlah Pelanggan")
    ax.set_ylabel("Segmen")
    st.pyplot(fig)
else:
    st.info("Tidak ada data transaksi pada rentang waktu yang dipilih.")