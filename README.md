# JCDS-2702 Final Project Beta Team

- Link Streamlit    : https://hotelbookingdemand-fjvezpfmcqzx9bsnspvcbg.streamlit.app
- Link Tableau      : https://public.tableau.com/app/profile/istyarahma.kusumastuti/viz/HotelBookingCancelationMonitoringDashboard/DashboardOK   

## Latar belakang dan Rumusan Masalah

Pembatalan booking yang berlangsung secara terus-menerus dapat berdampak pada:  
1. Hilangnya pendapatan karena kamar yang dibatalkan sulit terisi ulang di waktu dekat.  
2.Inefisiensi operasional karena pihak hotel tetap menyiapkan kamar dan layanan untuk tamu yang akhirnya tidak datang.  
3.Kesalahan strategi pemasaran jika tidak bisa membedakan tamu dengan risiko pembatalan tinggi atau rendah.  

Saat ini, hotel menggunakan strategi sederhana:  
1. **Overbooking** — menerima booking melebihi kapasitas untuk mengantisipasi pembatalan.\
Dengan cancellation rate sebesar 20% - 60% per hotel, maka overbooking dilakukan sangat wajar dilakukan dalam rangka memitigasi kemungkinan adanya booking cancellation. Biasanya, overbooking yang dilakukan berkisar antara 10% - 15% dari total kamar dari setiap hotel. Namun, konsekuensi dari dilakukannya strategi overbooking adalah apabila prediksi yang dilakukan tidak baik atau melakukan overbooking yang berlebihan, terdapat beberapa cost tambahan yang harus dikeluarkan management hotel, seperti relocation cost, social reputation, dan menurunnya revenue akibat besarnya diskon yang diberikan (Antonio et al., 2019).

Berikut merupakan rincian dari biaya yang berpotensi akan dikeluarkan apabila menerapkan strategi overbooking:
    - Biaya kompensasi: 180 EUR
    - Rellocation cost: 30 EUR
    - Complementary Discount: 100 EUR (Farbrother, 2021   
     - Reputation management: 3000 EUR (Clegg, 2024).

2. **First Come First Serve** — tidak melakukan prediksi, hanya menerima booking sesuai kapasitas.
*First come first serve* (FCFS) merupakan pendekatan yang umum digunakan oleh hotel dalam menagangi permintaan booking. Permintaan booking yang diajukan terlebih dulu akan diprioritaskan untuk memperoleh kamar terlebihi dahulu. Setiap booking harus menunggu hingga kamar tersedia, tanpa mempercepat masa inap tamu sebelumnya. Dalam FCFS tidak ada prioritas tambahan, seperti tamu VIP atau permintaan khusus lain, kecuali terdapat kebijakan khusus lainnya (Xu et al., 2014).


Kedua strategi ini memiliki kelemahan:  
1. **Overbooking** → risiko menolak tamu saat semua datang benar-benar hadir, menyebabkan ketidakpuasan.  
2. **First Come First Serve** → potensi kehilangan pendapatan karena kamar kosong akibat pembatalan mendadak (Hwang & Wen, 2009)

---

## Terminologi:
- **Booking**: pemesanan kamar atau akomodasi yang dilakukan oleh tamu hotel untuk periode waktu tertentu.
- **Overbooking**: situasi dimana sebuah hotel menerima lebih banyak reservasi daripada jumlah kamar yang sebenarnya tersedia.
- **First Come First Serve**: proses pemberian kamar dimana tamu yang datang lebih awal akan dilayani lebih dahulu sedangkan yang datang belakangan dilayani belakangan.
- **Deposit**: pembayaran atau penahanan dana yang dilakukan oleh tamu hotel sebagai jaminan atas penggunaan fasilitas hotel. Dalam konteks ini, deposit diberlakukan saat pemesanan untuk menanggulangi resiko kerugian apabila tamu batal datang.
- **No Deposit** → Tamu tidak diminta membayar deposit saat reservasi.  
- **Non Refund** → Tamu diminta membayar sejumlah deposit pada saat booking, dan uang ini **tidak dapat dikembalikan** jika tamu membatalkan.  
- **Refundable** → Tamu diminta membayar deposit, tetapi uangnya **bisa dikembalikan** jika tamu membatalkan sesuai aturan hotel.  

---

## Tujuan
### Tujuan Machine Learning
Membangun model Machine Learning untuk **memprediksi apakah sebuah booking berisiko dibatalkan**, sehingga hotel dapat:
- Mengidentifikasi **pelanggan mana saja yang berpotensi tinggi melakukan cancellation** sehingga dapat diberikan treatment khusus (seperti deposti lebih tinggi)

  

### Tujuan Bisnis
Membuat perangkat untuk membantu stakeholder **mengambil tindakan bisnis terbaik untuk meminimalisir kerugian dari pembatalan** menggunakan predictive approach atau menggunakan data analytics approach seperti:

#### Predictive Approach
- Mengoptimalkan strategi first come first serve.  
- Meminimalisir kehilangan pendapatan akibat kamar kosong.  
- Meningkatkan efisiensi operasional hotel.
#### Data Analytics Approach
- Menentukan**strategi deposit type** yang akan diberikan kepada customer berdasarkan probabilitas customer melakukan booking cancellation.
- Mengidentifikasi **tren transaksi bulanan hotel (2015 - 2017)** beserta variasinya pada periode **peak season, mid season, dan low season.**
- Mengidentifikasi **jenis-jenis kamar yang perlu dikurangi atau ditambah** kapasitas kamarnya untuk memitigasi dampak overbooking.
- Mendesain **strategi pricing berdasarkan detail pemesanan tertentu**, misalnya memberlakukan harga khusus untuk pemesanan yang dilakukan pada waktu-waktu tertentu (weekday/weekend atau musim-musim liburan).
- Menganalisis **profil customer** berdasarkan asal negara dan kategori market segment.

## Data Understanding

Dataset yang digunakan dalam project ini adalah *'Hotel Booking Demand'* yang merupakan adaptasi dari data artikel ilmiah yang dipublikasikan oleh Antonio et al dalam *Data in Brief* [(Antonio et al., 2019)](https://www.sciencedirect.com/science/article/pii/S2352340918315191). Adapun detail konteks dataset adalah sebagai berikut:

1. Dataset: Hotel Booking Demand
2. Sumber Data: [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data)
3. Jenis Data: Data Transaksi Booking Hotel
4. Tempat: Portugal, Lisbon
5. Periode Waktu: 2015 - 2017
6. Jumlah Variabel (Kolom): 32 Kolom
7. Jumlah Data (Baris): 119.390 Baris
8. Target: `is_canceled`


| Fitur                          | Tipe Data | Deskripsi                                                                       |
| ------------------------------ | --------- | ------------------------------------------------------------------------------- |
| hotel                          | object    | Jenis hotel: Resort Hotel (H1) atau City Hotel (H2).                            |
| lead_time                      | integer   | Jumlah hari antara tanggal pemesanan masuk ke sistem dengan tanggal kedatangan. |
| arrival_date_year              | integer   | Tahun kedatangan tamu.                                                          |
| arrival_date_month             | object    | Bulan kedatangan tamu.                                                          |
| arrival_date_week_number       | integer   | Nomor minggu dalam tahun pada saat kedatangan.                                  |
| arrival_date_day_of_month      | integer   | Tanggal kedatangan dalam satu bulan.                                            |
| stays_in_weekend_nights        | integer   | Jumlah malam akhir pekan (Sabtu/Minggu) yang dipesan atau diinapi.              |
| stays_in_week_nights           | integer   | Jumlah malam hari kerja (Senin–Jumat) yang dipesan atau diinapi.                |
| adults                         | integer   | Jumlah orang dewasa dalam pemesanan.                                            |
| children                       | float     | Jumlah anak-anak dalam pemesanan.                                               |
| babies                         | integer   | Jumlah bayi dalam pemesanan.                                                    |
| meal                           | object    | Jenis paket makanan yang dipesan.                                               |
| country                        | object    | Negara asal tamu.                                                               |
| market_segment                 | object    | Segmen pasar dari mana pemesanan berasal (misalnya Online, Offline, dll).       |
| distribution_channel           | object    | Saluran distribusi pemesanan (misalnya GDS, TA/TO, dll).                        |
| is_repeated_guest              | integer   | Apakah tamu merupakan pelanggan berulang (1) atau tidak (0).                    |
| previous_cancellations         | integer   | Jumlah pemesanan sebelumnya yang dibatalkan oleh tamu.                          |
| previous_bookings_not_canceled | integer   | Jumlah pemesanan sebelumnya yang tidak dibatalkan oleh tamu.                    |
| reserved_room_type             | object    | Tipe kamar yang awalnya dipesan.                                                |
| assigned_room_type             | object    | Tipe kamar yang akhirnya diberikan.                                             |
| booking_changes                | integer   | Jumlah perubahan yang dilakukan pada pemesanan.                                 |
| deposit_type                   | object    | Jenis deposit (misalnya No Deposit, Non Refund, Refundable).                    |
| agent                          | float     | ID agen perjalanan (jika pemesanan dilakukan via agen).                         |
| company                        | float     | ID perusahaan (jika pemesanan dilakukan via perusahaan).                        |
| days_in_waiting_list           | integer   | Jumlah hari pemesanan berada di waiting list.                                   |
| customer_type                  | object    | Jenis pelanggan (Contract, Group, Transient, Transient-Party).                  |
| adr                            | float     | Average Daily Rate, rata-rata harga kamar per malam.                            |
| required_car_parking_spaces    | integer   | Jumlah slot parkir yang diminta.                                                |
| total_of_special_requests      | integer   | Jumlah permintaan khusus dari tamu.                                             |
| reservation_status             | object    | Status akhir reservasi (Canceled, Check-Out, No-Show).                          |
| reservation_status_date        | object    | Tanggal ketika status reservasi terakhir diperbarui.                            |



| Target                          | Tipe Data | Deskripsi                                                                       |
| ------------------------------ | --------- | ------------------------------------------------------------------------------- |
| is_canceled                    | integer   | Penanda apakah pemesanan dibatalkan (1) atau tidak (0).


## Data Cleaning:

Tambah Kolom

1. Room Sesuai/Tidak antara reserve dan assign
2. Kolom Arrival Date
3. Kolom Country (PRT vs Foreign)
4. Kolom Deposit_No Deposit berdasarkan ada tidaknya Deposit


Hapus Outlier
ADR >  €1000 

Missing Data

Imputation:
Agent
Company

Median:
Country
Children

Duplicate
Action: keep

## Exploratory Data Analysis:

Deposit Type
Lead Time
Market Segment
Distribution Channel
Customer Type
Previous Cancellations
Special Requests
Required Car Parking Space
Room Details
Room Type
Booking traffic per season
 Average Daily Rate
Country
Portugal vs Foreigner

## Define X & Y dan Data Splitting

### Define X & Y

Sebelum mendefinisikan X dan y, perlu dilakukan pengubahan tipe data pada kolom agent dan company. Karena untuk menghindari model membacanya sebagai numerikal.

Kemudian, ada beberapa kolom yang tidak kami masukkan sebagai fitur karena beberapa alasan, yaitu : 
- assigned_room_type : Model Machine Learning digunakan saat pemesanan berlangsung. Sehingga kesesuaian tipe kamar yang didapatkan belum diketahui secara pasti.
- is_canceled : target model
- deposit_type : Tidak kami masukkan karena setelah dilakukan prediksi pembatalan pemesanan menggunakan model machine learning, baru pihak hotel yang menentukan tipe depositnya apa.
- room_sesuai : Ini adalah kolom baru yang berasal dari assigned_room_type dan reserved_room_type. Maka nilai dari room_sesuai juga belum diketahui saat melakukan prediksi.
- arrival date: Karena merupakan kolom buatan yang berasal dari beberapa kolom lain. Ini bertujuan untuk menghindari redundant atau multicolinearity.
- country_prt : Merupakan kolom baru yang berasal dari kolom country, sehingga cukup menggunakan kolom country saja.
- Deposit_No_Deposit : Merupakan kolom baru yang berasal dari kolom deposit_type.
- reservation_status : status terbaru dari pemesanan belum diketahui, dan kolom ini juga terdapat redundant informasi dengan kolom is_canceled yang mana merupakan target dari model yang akan dibuat.
- reservation_status_date : tanggalnya belum diketahui ketika pemesanan sedang berlangsung.

### Data Splitting 


random_state = 42(sebenarnya bebas, namun umumnya nilai yang digunakan kebanyakan orang adalah 42), 
stratify=y bertujuan untuk menyelaraskan proprosi antar kelas saat dilakukan splitting. Dan test_size=0.2 untuk membagi data menjadi data train 80% dan data test 20% (ini adalah proporsi pembagian data yang sering digunakan).

## Data Preprocessing

Ada 3 pipeline yang dibuat karena ada beberapa kolom yang memerlukan dua kali preprocessing, yaitu : 

- FillnaEncoding : agent dan company. Dua kolom ini memerlukan impute missing value terlebih dahulu, dan sifatnya kategorial. Kemudian dilakukan binary encoding, karena sifatnya kategorial dan nilai unik dari dua kolom ini terlalu banyak, sehingga kurang cocok untuk menggunakan one hot encoder, dan kurang cocok menggunakan ordinal encoding (karena tidak ada tingkatannya)
- ImputeScaling : children. Kolom ini terdapat misisng value, sehingga diperlukan impute menggunakan median (karena tidak terdistribusi normal) dan setelah itu dilakukan scaling menggunakan robust scaler (tidak terdistribusi normal).
- imputeEncoding : country. Terdapat missing value, sehingga perlu diimpute menggunakan modus (karena kategorial) dan binary encoding digunakan karena kolom country memiliki nilai unik yang banyak.

--- 

- Standard_Scaling: booking_changes
- Min_Max_Scaling: arrival_date_year, arrival_date_week_number, arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights, children, babies, previous_cancellations, previous_bookings_not_canceled, required_car_parking_spaces, total_of_special_requests
- Robust_Scaling: lead_time, adult, days_in_waiting_list, adr


- OneHotEncoding: meal, distribution_channel, customer_type, hotel (nilai unik sedikit)
- BinaryEncoding: arrival_date_month, market_segment, reserved_room_type (nilai unik banyak)
- Passthrough: is_repeated_guest (sudah berisi nilai 0 dan 1)

## Cross Validation

logreg = LogisticRegression
knn = KNeighborsClassifier
tree = DecisionTreeClassifier
rf = RandomForestClassifier
ada = AdaBoostClassifier
gbc = GradientBoostingClassifier
xgbc = XGBClassifier
Berdasarkan hasil cross-validasi tersebut, didapatkan bahwa algoritma XGBoost yang menggunakan metode resampling dengan RandomOverSampler mendapatkan rata-rata recall tertinggi yang mencapai 86,85% dan bisa dibilang cukup stabil dengan standard deviasi sekitar 0,0034

## Feature Selection

Setelah melakukan Feature Selection, kami memutuskan yang awalnya menggunakan 25 fitur (sebelum preprocessing), kami hanya menggunakan 18 fitur (sebelum preprocessing) saja yang akan digunakan. 18 fitur ini adalah fitur-fitur yang memiliki feature_importance tertinggi.

Maka, dari itu kami mendefinisikan X dan y, melakukan splitting data, preprocessing, dan modeling kembali untuk menggunakan 18 fitur saja.

## Hasil Utama

XGBoost Base Model dengan threshold=0,56.

Fitur dominan:
1. required_car_parking_spaces
2. total_of_special_requests
3. previous_cancellations
4. lead_time
5. customer_type_Transient
6. arrival_date_year
7. customer_type_Transient-Party
8. adr (Average Daily Rate / harga kamar)
9. booking_changes
10. Previous_bookings_not_canceled

## Conclusion and Recommendation

PERBANDINGAN SEBELUM VS SEDUDAH MENGGUNAKAN MACHINE LEARNING

Sebelum Menggunakan Machine Learning : 3.541.700 euro
Setelah Menggunakan Machine Learning : 3.156.585,6 euro

Selisih biaya yang berhasil dihemat : 385.114,4 euro / Rp 7.478.035.884,88 atau Rp 7,48 Miliar

### Komposisi Segmen Pelanggan:

Melalui analisis, yang telah dilakukan, dapat disimpulkan bahwa terdapat beberapa karakteristik pemesan yang memiliki kemungkinan pembatalan pemesanan kamar lebih tinggi:

1. Deposit

Berdasarkan tipe deposit yang dibayar oleh pemesan, terdapat tiga tipe deposit:
No Deposit, Refundable dan Non-Refundable
  - Pembatalan terbanyak terdapat pada tipe Non-Refundable, dengan 99.4% dari pemesan dengan deposit jenis ini melakukan pembatalan.

Karena itu, perhatian khusus perlu diberikan pada tipe deposit Non-Refundable.

2. Country

Berdasarkan asal negara, Tamu terbanyak dari grup hotel ini berasal dari Portugal, sebanyak 45.490 tamu (40,7%). Namun, apabila data hanya dikelompokkan antara tamu Portugal dan Foreign Country, maka tamu yang berasal dari mancanegara memiliki frekuensi yang sedikit lebih banyak, yaitu 70.800 tamu (59,3%).
- Akan tetapi, segmen tamu Portugal mencerminkan permasalahan yang signifikan, yaitu tingkat pembatalan pemesanan tamu Portugal mencapai 56,6%.
- Berdasarkan market segment, segmen group menunjukkan kemungkinan pembatalan paling banyak, dengan 61,1% diikuti dengan Online TA serta Offline TA/TO memiliki proporsi yang hampir serupa dengan proporsi pembatalan masing-masing 36.7% dan 34.3%

3. Customer Type

Berdasarkan customer type,  **Transient**, proporsi pembatalan menunjukkan persentase yang paling signifikan, yaitu 40% disusul dengan **Transient-Party**, dengan persentase 31%

4. Lead Time

Berdasarkan lead time, ditemukan insight yang menarik yaitu:
pemesan dengan Lead Time yang **lebih lama** (dengan Mean 145) cenderung berakhir membatalkan pemesanannya.

### Hasil Analisis Detail Pemesanan:

Selain itu, berdasarkan analisis, kami mendapatkan insight penting mengenai tipe kamar yang menyumbang paling banyak pemesanan dan juga berpengaruh terhadap cost dari segi pembatalan.

- Room Type A adalah tipe kamar dengan pemesanan terbanyak, baik yang dibatalkan maupun yang tidak, mencakup sebagian besar distribusi data. Dari 85,993 pemesanan (terbanyak dibandingkan pemesanan kamar lain) 33,630 melakukan pembatalan, sehingga mencakup 39,1% dari total pemesanan. Selaras dengan volume pemesanan yang lebih besar, Room Type A juga menyumbang jumlah pembatalan terbesar secara absolut.


Selain itu, terdapat juga beberapa insight tentang detail pemesanan yang dapat membantu strategi pricing berdasarkan detail pemesanan tertentu, tepatnya pada waktu pemesanan tertentu:
1. **Peak Season (Musim Panas: Juni - Agustus)**
  Secara konsisten mencatat volume transaksi yang tinggi, dengan puncak tertinggi terjadi pada Juni 2017 (6.313 transaksi).
2. **Mid Season (Musim Semi dan Gugur: Maret - Mei dan September - Oktober)**
  Merupakan periode dengan jumlah transaksi terbesar, yaitu 54.342 transaksi. Volume yang cukup besar tersebut merupakan hasil akumulasi dari season yang lebih panjang dibanding season lainnya (10 bulan).
  
Berdasarkan analisis di atas, kami memutuskan untuk memberi perhatian lebih banyak terhadap Mid Season karena volume transaksinya yang besar dan jangka waktunya yang lama.


Kesimpulan di atas juga didukung oleh analisis mendalam pada volume pemesanan kamar hotel per musim, dengan pembagian jenis hotel untuk mendukung analisis:
1. Pada Mid-Season dimana transaksi terjadi paling tinggi, pemesanan didominasi oleh kamar A baik di City Hotel, yaitu 29.561 pemesanan dan Resort Hotel yaitu 10.310 pemesanan
2. Namun demikian, city Hotel secara konsisten memiliki jumlah pemesanan lebih tinggi daripada Resort Hotel, terutama untuk Room Type A yang jumlahnya 2–3 kali lipat lebih banyak di City Hotel.

Karena itu, segi pengambilan keputusan bisnis, terutama manajemen kapasitas, dapat dipusatkan di **City Hotel** pada **tipe room A**, terutama pada **Mid-Season** karena transaksi volumenya yang tinggi dan signifikan.

### Kesimpulan Akhir
- Tipe Deposit Non-Refundable menunjukkan proporsi pembatalan paling besar, karena itu keputusan bisnis akan terfokus pada tipe deposit ini
Detail segmen pelanggan lainnya yang mungkin berpengaruh terhadap pembatalan adalah
- Asal negara (Portugal paling banyak)
Customer type Transient dan Transient-Party
Lead Time yang lama
- Berdasarkan tipe kamar, Room Type A adalah tipe kamar dengan pemesanan terbanyak, baik yang dibatalkan maupun yang tidak. Karena itu, strategi manajemen kapasitas kamar akan dipusatkan pada tipe Room A, terutama di City Hotel
- Berdasarkan waktu pemesanan, Mid Season (Musim Semi dan Gugur: Maret - Mei dan September - Oktober)** merupakan periode dengan jumlah transaksi terbesar, yaitu 54.342 transaksi dengan catatan memiliki season lebih panjang (10 bulan). Namun, karena kontribusinya terhadap pemesanan signifikan dan konsisten, maka Mid Season juga dapat dipertimbangkan dalam pengambilan keputusan bisnis.

### Rekomendasi Strategi
- Booking dengan lead time panjang → perlu deposit / penalti cancel.
- Customer transient & transient-party → konfirmasi tambahan / syarat cancel lebih ketat.
- Tamu dengan banyak special requests → bisa diprioritaskan, karena lebih serius.
- Tamu dengan riwayat cancel tinggi → batasi fleksibilitas (misalnya non-refundable).
- Harga tinggi (adr) → tawarkan opsi fleksibel seperti reschedule agar tidak langsung cancel.

# Kontributor
- Pradna Aqmaril Paramitha
- Istyarahma Kansya Kusumastuti
- Sulthanur Iman Fatahillah
