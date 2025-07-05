# 🚀 Panduan Penggunaan Revolutionary AGI Forex Trading System

Dokumen ini berisi panduan lengkap untuk menggunakan Revolutionary AGI Forex Trading System dengan 5 teknologi jenius yang tidak terkalahkan.

## 📋 Daftar Isi

1. [Persiapan Awal](#persiapan-awal)
2. [Konfigurasi Sistem](#konfigurasi-sistem)
3. [Menjalankan Sistem](#menjalankan-sistem)
4. [Menggunakan Telegram Bot](#menggunakan-telegram-bot)
5. [Mengakses API](#mengakses-api)
6. [Melakukan Backtest](#melakukan-backtest)
7. [Mode Demo](#mode-demo)
8. [Pemecahan Masalah](#pemecahan-masalah)

## 🛠️ Persiapan Awal

### Prasyarat

Sebelum menggunakan sistem, pastikan Anda memiliki:

- Python 3.8 atau lebih baru
- pip (Python package manager)
- Git (untuk clone repository)
- Koneksi internet stabil
- API key dari broker forex (OANDA, FXCM, dll)
- Token bot Telegram (opsional, untuk notifikasi)

### Instalasi

1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/revolutionary-agi-forex.git
   cd revolutionary-agi-forex
   ```

2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

3. Salin file `.env.example` ke `.env`:
   ```bash
   cp .env.example .env
   ```

4. Edit file `.env` dengan editor teks pilihan Anda dan isi dengan API key dan kredensial yang diperlukan.

## ⚙️ Konfigurasi Sistem

### File Konfigurasi

Sistem menggunakan file konfigurasi YAML yang terletak di `config/config.yaml`. File ini berisi semua parameter yang diperlukan untuk menjalankan sistem.

### Parameter Penting

Berikut beberapa parameter penting yang perlu dikonfigurasi:

#### 1. API Keys

```yaml
# Data Collection Configuration
data_collection:
  providers:
    oanda:
      enabled: true
      api_key: "${OANDA_API_KEY}"
      account_id: "${OANDA_ACCOUNT_ID}"
```

#### 2. Pasangan Mata Uang

```yaml
# Currency pairs to monitor
currency_pairs:
  major:
    - "EURUSD"
    - "GBPUSD"
    - "USDJPY"
```

#### 3. Timeframes

```yaml
# Timeframes to collect
timeframes:
  - "M1"   # 1 minute
  - "M5"   # 5 minutes
  - "M15"  # 15 minutes
  - "H1"   # 1 hour
  - "H4"   # 4 hours
  - "D1"   # Daily
```

#### 4. Manajemen Risiko

```yaml
# Risk Management Configuration
risk_management:
  position_sizing:
    method: "kelly_criterion"
    base_risk_per_trade: 0.02  # 2% of account
```

#### 5. Konfigurasi Telegram

```yaml
# Telegram Bot Configuration
telegram:
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  webhook_url: "${TELEGRAM_WEBHOOK_URL}"
```

## 🚀 Menjalankan Sistem

### Menjalankan Sistem Lengkap

Untuk menjalankan sistem lengkap dengan semua fitur:

```bash
python run_revolutionary.py
```

Opsi tambahan:
- `--config`: Menentukan file konfigurasi kustom
- `--verbose`: Mengaktifkan logging verbose
- `--demo`: Menjalankan dalam mode demo (tidak melakukan trading sungguhan)

Contoh:
```bash
python run_revolutionary.py --config custom_config.yaml --verbose
```

### Menjalankan Komponen Terpisah

#### 1. API Server

Untuk menjalankan hanya API server:

```bash
python run_api_server.py --port 8000
```

#### 2. Telegram Bot

Untuk menjalankan hanya Telegram bot:

```bash
python run_telegram_bot.py
```

Untuk mode webhook:
```bash
python run_telegram_bot.py --webhook
```

#### 3. Mode Demo

Untuk menjalankan sistem dalam mode demo (tanpa trading sungguhan):

```bash
python run_demo.py
```

## 📱 Menggunakan Telegram Bot

### Memulai Bot

1. Buat bot Telegram baru melalui [@BotFather](https://t.me/BotFather)
2. Dapatkan token bot dan masukkan ke file `.env`
3. Jalankan bot dengan perintah:
   ```bash
   python run_telegram_bot.py
   ```
4. Mulai percakapan dengan bot Anda di Telegram

### Perintah Bot

Bot mendukung perintah-perintah berikut:

- `/start` - Memulai bot dan menampilkan pesan selamat datang
- `/help` - Menampilkan bantuan dan daftar perintah
- `/status` - Menampilkan status sistem
- `/signals` - Menampilkan sinyal trading terbaru
- `/performance` - Menampilkan laporan kinerja
- `/settings` - Mengubah pengaturan notifikasi
- `/subscribe` - Berlangganan notifikasi sinyal
- `/unsubscribe` - Berhenti berlangganan notifikasi

### Notifikasi

Bot akan mengirimkan notifikasi untuk:

- Sinyal trading baru
- Eksekusi trading
- Penutupan posisi
- Peringatan risiko
- Berita ekonomi penting
- Status sistem

## 🌐 Mengakses API

### Endpoint API

API dapat diakses di `http://localhost:8000/api/v2/` dengan endpoint berikut:

- `/revolutionary-status` - Status sistem
- `/quantum-performance` - Performa quantum optimizer
- `/swarm-intelligence` - Status swarm intelligence
- `/computer-vision-analysis` - Analisis computer vision
- `/blockchain-verification` - Status verifikasi blockchain
- `/neuro-economic-pulse` - Pulse ekonomi real-time
- `/docs` - Dokumentasi API interaktif

### Autentikasi API

API menggunakan autentikasi API key. Tambahkan header berikut ke semua permintaan:

```
X-API-Key: your_api_key_here
```

API key dapat dikonfigurasi di file `.env`.

### Contoh Penggunaan API

#### Mendapatkan Status Sistem

```bash
curl -X GET "http://localhost:8000/api/v2/revolutionary-status" -H "X-API-Key: your_api_key_here"
```

#### Mendapatkan Sinyal Trading Terbaru

```bash
curl -X GET "http://localhost:8000/api/v2/signals/latest" -H "X-API-Key: your_api_key_here"
```

## 📊 Melakukan Backtest

### Menjalankan Backtest

Untuk menjalankan backtest pada data historis:

```bash
python run_backtest.py --start-date 2023-01-01 --end-date 2023-12-31 --pairs EURUSD GBPUSD --timeframes H1 D1
```

Opsi tambahan:
- `--strategies`: Strategi yang akan diuji (default: semua)
- `--report`: Path untuk menyimpan laporan backtest
- `--verbose`: Mengaktifkan logging verbose

### Laporan Backtest

Laporan backtest akan disimpan dalam format Markdown dan berisi:

- Metrik kinerja (win rate, profit factor, Sharpe ratio, dll)
- Grafik kinerja
- Daftar trade
- Analisis drawdown
- Statistik per pasangan mata uang dan timeframe

## 🎮 Mode Demo

### Menjalankan Mode Demo

Mode demo memungkinkan Anda menjalankan sistem tanpa melakukan trading sungguhan:

```bash
python run_demo.py
```

### Fitur Mode Demo

- Simulasi trading dengan data pasar real-time
- Semua 5 teknologi jenius berfungsi penuh
- Tidak ada trading sungguhan yang dilakukan
- Laporan kinerja simulasi
- Visualisasi sinyal trading

## 🔧 Pemecahan Masalah

### Masalah Umum

#### 1. Kesalahan API Key

**Gejala**: Pesan error "Invalid API key" atau "Authentication failed"

**Solusi**:
- Periksa API key di file `.env`
- Pastikan API key masih aktif
- Periksa apakah Anda menggunakan environment yang benar (demo/live)

#### 2. Kesalahan Database

**Gejala**: Pesan error "Database connection failed"

**Solusi**:
- Periksa kredensial database di file `.env`
- Pastikan database server berjalan
- Periksa apakah skema database sudah dibuat

#### 3. Bot Telegram Tidak Merespons

**Gejala**: Bot Telegram tidak merespons perintah

**Solusi**:
- Periksa token bot di file `.env`
- Pastikan bot sedang berjalan (`python run_telegram_bot.py`)
- Periksa log untuk error
- Restart bot

#### 4. Sistem Tidak Menghasilkan Sinyal

**Gejala**: Tidak ada sinyal trading yang dihasilkan

**Solusi**:
- Periksa konfigurasi pasangan mata uang dan timeframe
- Pastikan data pasar tersedia
- Periksa parameter confidence threshold
- Periksa log untuk warning atau error

### Log dan Debugging

Log sistem tersimpan di direktori `logs/`. Untuk debugging lebih lanjut, jalankan sistem dengan flag `--verbose`:

```bash
python run_revolutionary.py --verbose
```

## 📞 Dukungan

Jika Anda mengalami masalah yang tidak tercantum di sini, silakan hubungi dukungan:

- Email: support@revolutionary-agi-forex.com
- Telegram: @RevolutionaryAGIForex
- GitHub Issues: [https://github.com/yourusername/revolutionary-agi-forex/issues](https://github.com/yourusername/revolutionary-agi-forex/issues)

---

🚀 **Selamat menggunakan Revolutionary AGI Forex Trading System!** 🚀

*"The future of trading is here, and it's revolutionary!"* - Revolutionary AGI Systems