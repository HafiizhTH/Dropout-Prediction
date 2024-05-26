import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set halaman streamlit
st.set_page_config(
    page_title="Dropout Prediction",
    page_icon=":bar_chart:",
    layout="wide"
)

def generate_palette(series):
    max_val = series.value_counts().idxmax()
    min_val = series.value_counts().idxmin()
    colors = ['#DD5746' if val == min_val else '#FFC470' if val == max_val else '#4793AF' for val in series.unique()]
    return dict(zip(series.unique(), colors))

# Title and description
st.title("Dropout Prediction")
st.write("adalah sebuah aplikasi model yang bertujuan Memprediksi serta Menganalisis siswa yang terindikasi keluar dari Perguruan Tinggi. Sehingga pihak kampus dapat mengambil kebijakan dan mengevalusi permasalahan tersebut.")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Prediksi", "Informasi Siswa", "FAQ"])

# Initialize session state to hold the uploaded data
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# Example file data
dataset_url = "https://raw.githubusercontent.com/HafiizhTH/Dropout-Prediction/main/Data/Data_to_Visualisasi.csv"
df_sample = pd.read_csv(dataset_url)
df_sample = df_sample.sample(20)

# Convert DataFrame ke file CSV
csv = df_sample.to_csv(index=False)

# Fitur yang digunakan pada saat model
model_features = ['Marital_status', 'Course', 'Daytime_evening_attendance', 'Mothers_occupation', 'Fathers_occupation', 'Admission_grade',
                  'Displaced', 'Educational_special_needs', 'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
                  'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
                  'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_enrolled',
                  'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade']

# Halaman Prediksi
if page == "Prediksi":
    st.header("Mulai Prediksi!")
    st.write("Pilih metode prediksi: single data (input manual) atau multiple data (upload dataset)")

    # Membuat tab untuk single prediction dan multi-prediction
    tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

    # Bagian Single-predict
    with tab1:
        st.write("Silakan masukkan data siswa untuk diprediksi:")
        
        # Definisi pilihan untuk setiap kolom
        predefined_options = {
            'Education': {'Belum Menikah': 1, 'Sudah Menikah': 2, 'Duda': 3, 'Berpisah': 4, 'Belum Menikah': 5, 'Berpisah': 6},
            'Course': {'Biofuel Production Technologies': 33, 'Animation and Multimedia Design': 171, 
                       'Social Service (evening attendance)': 8014, 'Agronomy': 9003, 'Communication Design': 9070, 
                       'Veterinary Nursing': 9085, 'Informatics Engineering': 9119, 'Equinculture': 9130, 'Management': 9147, 
                       'Social Service': 9238, 'Tourism': 9254, 'Nursing': 9500, 'Oral Hygiene': 9556, 
                       'Advertising and Marketing Management': 9670, 'Journalism and Communication': 9773, 
                       'Basic Education': 9853, 'Management': 9991},
            'Daytime_evening_attendance': {'Evening': 0, 'Daytime': 1},
            'Mothers_occupation': {'Belum/Tidak Bekerja': [0, 90, 99], 'Direktur/Manager': [1, 112, 114], 
                                   'Spesialis Pendidikan': [2, 121, 123], 
                                   'Spesialis IT/Teknisi': [3, 8, 131, 132, 135, 154, 171, 174, 181], 
                                   'Spesialis Kesehatan': [122, 153], 'Spesialis Keuangan': [124, 143], 'Spesialis Hukum': 134, 
                                   'Manajemen/Administrasi': [4, 141, 144], 'Layanan dan Jasa': [5, 151, 152, 175, 182, 194, 195], 
                                   'Petani/Perikanan/Kehutanan': [6, 161, 163, 192], 'Konstruksi/Transportasi': [7, 183, 193], 
                                   'Tentara': [10, 101, 102, 103]},
            'Fathers_occupation': {'teks': 0},
            'Displaced': {'Tidak': 0, 'Iya': 1},
            'Educational_special_needs': {'Tidak': 0, 'Iya': 1},
            'Debtor': {'Tidak': 0, 'Iya': 1},
            'Tuition_fees_up_to_date': {'Tidak': 0, 'Iya': 1},
            'Gender': {'Perempuan': 0, 'Laki-laki': 1},
            'Scholarship_holder': {'Tidak': 0, 'Iya': 1},
            'International': {'Tidak': 0, 'Iya': 1}
        }

        # Input fields
        user_input = {}
        col1, col2, col3 = st.columns(3)
        for i, feature in enumerate(model_features):
            column = col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3
            if feature in predefined_options:
                user_input[feature] = column.selectbox(f"{feature}", predefined_options[feature])
            else:
                user_input[feature] = column.number_input(f"{feature}", min_value=0, max_value=200)

        # Predict button for single data
        if st.button("Predict Single Data", key="predict_single"):
            try:
                # Load model dari GitHub
                filename = 'https://raw.githubusercontent.com/HafiizhTH/Dropout-Prediction/main/Data/result_model.pkl'
                response = requests.get(filename)
                if response.status_code == 200:
                    model = pickle.loads(response.content)
                else:
                    st.error("Gagal memuat model. Status code:", response.status_code)
                    st.stop()

                # Buat Datafreame
                user_data = pd.DataFrame([user_input])

                # Preprocessing
                num_features = user_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                cat_features = user_data.select_dtypes(include=['object', 'category']).columns.tolist()

                num_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])

                cat_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder())
                ])

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', num_pipeline, num_features),
                        ('cat', cat_pipeline, cat_features)
                    ],
                    remainder='passthrough'
                )

                user_data_processed = preprocessor.fit_transform(user_data)
                prediction = model.predict(user_data_processed)

                if prediction[0] == 1:
                    st.success("siswa ini diprediksi mengalami dropout.")
                else:
                    st.success("siswa ini diprediksi tidak mengalami dropout.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Bagian Multi-predict
    with tab2:
        st.write("Upload file siswa dropout dalam format .csv atau .xlsx")

        # Download example file
        st.download_button(
            label="Download Example File",
            data=csv,
            file_name="example_student_dataset.csv",
            mime='text/csv',
            key="download_example"
        )
        
        # Upload file
        uploaded_file = st.file_uploader("Upload file Anda", type=['csv', 'xlsx'], help="Batas 200MB per file • CSV, XLSX", key="uploader")
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                    
                if 'Status' in df.columns:
                    df = df.drop(columns=['Status'])
                
                if df.empty or df.shape[0] < 1:
                    st.error("Dataset kosong. Pastikan dataset memiliki minimal 1 baris.")
                else:
                    st.session_state.uploaded_data = df
                    st.success("File berhasil diupload.")
                
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error reading the file: {e}")
        
        if st.button("Predict", key="predict_button"):
            if st.session_state.uploaded_data is not None:
                st.info("Proses prediksi dimulai.")
                df = st.session_state.uploaded_data
                
                try:
                    filename = 'https://raw.githubusercontent.com/HafiizhTH/Dropout-Prediction/main/Data/result_model.pkl'
                    response = requests.get(filename)
                    if response.status_code == 200:
                        model = pickle.loads(response.content)
                    else:
                        st.error("Gagal memuat model. Status code:", response.status_code)
                        st.stop()
                    
                    missing_features = set(model_features) - set(df.columns)
                    if missing_features:
                        st.warning("Beberapa fitur yang dibutuhkan oleh model tidak ditemukan dalam dataset:")
                        st.write(missing_features)
                        st.stop()
                    
                    df = df[model_features]

                    num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

                    num_pipeline = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ])

                    cat_pipeline = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('ordinal_encoder', OrdinalEncoder())
                    ])

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', num_pipeline, num_features),
                            ('cat', cat_pipeline, cat_features)
                        ],
                        remainder='passthrough'
                    )
                    
                    df_processed = preprocessor.fit_transform(df)
                    predictions = model.predict(df_processed)
                    df['Dropout_predict'] = predictions
                    
                    dropout_student = df[df['Dropout_predict'] == 1]
                    no_dropout_student = df[df['Dropout_predict'] == 0]
                    
                    st.subheader("Hasil Prediksi")
                    st.write(f"Jumlah siswa yang diprediksi mengalami dropout: {len(dropout_student)}")
                    st.write(f"Jumlah siswa yang diprediksi tidak mengalami dropout: {len(no_dropout_student)}")
                    
                    total_student = len(df)
                    dropout_percentage = (len(dropout_student) / total_student) * 100
                    st.write(f"Persentase siswa yang diprediksi mengalami dropout: {dropout_percentage:.0f}" "%")

                    if not dropout_student.empty:
                        st.subheader("siswa yang diprediksi akan mengalami dropout:")
                        st.dataframe(dropout_student)
                    else:
                        st.info("Tidak terdapat siswa yang mengalami dropout.")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Error downloading the model: {e}")
                except pickle.UnpicklingError as e:
                    st.error(f"Error unpickling the model: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
            else:
                st.warning("Silakan upload file terlebih dahulu.")
                
# Halaman Informasi siswa
elif page == "Informasi Siswa":
    st.header("Informasi Siswa")
    
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        
        tab1, tab2 = st.tabs(["Data Deskriptif", "Data Visualisasi"])
        
        with tab1:
            st.subheader("Tampilan dari dataset")
            
            max_rows = len(df)
            num_rows = st.number_input("Jumlah baris yang ditampilkan", min_value=1, max_value=max_rows, value=min(5, max_rows))
            
            st.dataframe(df.head(num_rows))
            st.subheader("Informasi Kolom")
            st.write(df.describe(include='all').transpose())
        
        with tab2:
            st.subheader("Visualisasi Data")
            
            select_col = st.selectbox("Pilih kolom untuk divisualisasikan", df.columns)
            
            if pd.api.types.is_numeric_dtype(df[select_col]):
                st.write(f"Visualisasi Histogram untuk kolom {select_col}")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(df[select_col], ax=ax, color='#4793AF')
                
                st.pyplot(fig)
            else:
                st.write(f"Visualisasi Bar Chart untuk kolom {select_col}")
                fig, ax = plt.subplots(figsize=(10, 5))
                palette = generate_palette(df[select_col])
                sns.countplot(y=df[select_col], palette=palette, ax=ax)
                
                # Menambahkan anotasi jumlah siswa dengan jarak
                for container in ax.containers:
                    ax.bar_label(container, label_type='edge', padding=5, fontsize=10, color='black', fontweight='bold')
                
                st.pyplot(fig)
                
    else:
        st.info("Upload dataset pada tab Prediksi untuk melihat kontennya di sini.")

# Halaman FAQ
elif page == "FAQ":
    st.header("Frequently Asked Questions (FAQ)")

    with st.expander("Apa itu dropout Prediction?"):
        st.write("""
        dropout Prediction adalah sebuah aplikasi model yang bertujuan untuk memprediksi apakah seorang siswa berisiko akan berhenti kuliah atau tidak.
        """)

    with st.expander("Kolom yang digunakan untuk model"):
        st.write("""
        Berikut adalah kolom yang harus ada ketika upload file untuk model prediksi dropout:
        - Marital_status 
        - Course
        - Daytime_evening_attendance
        - Mothers_occupation
        - Fathers_occupation 
        - Admission_grade
        - Displaced
        - Educational_special_needs
        - Debtor
        - Tuition_fees_up_to_date
        - Gender
        - Scholarship_holder
        - Age_at_enrollment
        - International
        - Curricular_units_1st_sem_enrolled
        - Curricular_units_1st_sem_evaluations
        - Curricular_units_1st_sem_approved
        - Curricular_units_1st_sem_grade
        - Curricular_units_2nd_sem_enrolled
        - Curricular_units_2nd_sem_approved
        - Curricular_units_2nd_sem_grade   
        """)

    with st.expander("Bagaimana cara menggunakan aplikasi ini?"):
        st.write("""
        Anda dapat menggunakan aplikasi ini dengan dua cara:
        1. Single Predict: Memasukkan data secara manual untuk satu siswa dan mendapatkan prediksi.
        2. Multi Predict: Mengupload file dataset berisi data beberapa siswa dan mendapatkan prediksi untuk semua siswa dalam file tersebut.
        """)

    with st.expander("Apa yang harus dilakukan jika terdapat error?"):
        st.write("""
        Jika Anda mengalami error, pastikan format data yang Anda masukkan sudah benar dan sesuai dengan kolom yang dibutuhkan. Jika error masih terjadi, Anda bisa menghubungi saya untuk bantuan lebih lanjut.
        
        Email: hafizhjunior54@gmail.com
        """)

    with st.expander("Apa keuntungan menggunakan aplikasi ini?"):
        st.write("""
        Aplikasi ini membantu Pihak Perguruan Tinggi dalam mengidentifikasi siswa yang terindikasi keluar atau berhenti kuliah. Sehingga Pihak Perguruan Tinggi dapat mengambil tindakan pencegahan yang diperlukan untuk mempertahankan siswa tersebut.
        """)
