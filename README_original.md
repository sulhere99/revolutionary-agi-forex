# 🤖 AGI Forex Trading System

Sistem AGI (Artificial General Intelligence) untuk trading forex yang sangat canggih dengan kemampuan self-learning, analisis real-time, dan integrasi Telegram yang komprehensif.

## 🌟 Fitur Utama

### 🧠 AGI Brain
- **Self-Learning AI**: Model yang dapat belajar dan memperbaiki diri sendiri tanpa intervensi manusia
- **Multi-Model Ensemble**: Kombinasi Neural Networks, XGBoost, LightGBM, dan Reinforcement Learning
- **Real-time Analysis**: Analisis pasar real-time dengan multiple timeframes
- **Adaptive Strategy**: Strategi yang beradaptasi dengan kondisi pasar yang berubah
- **Pattern Recognition**: Deteksi pola candlestick dan formasi teknikal

### 📊 Advanced Market Analysis
- **Technical Analysis**: 50+ indikator teknikal dengan optimasi otomatis
- **Fundamental Analysis**: Integrasi kalender ekonomi dan analisis berita
- **Sentiment Analysis**: Analisis sentiment dari news dan social media
- **Market Regime Detection**: Deteksi otomatis regime pasar (trending, ranging, volatile)
- **Support/Resistance**: Identifikasi level support dan resistance dinamis

### ⚠️ Risk Management
- **Dynamic Position Sizing**: Menggunakan Kelly Criterion dan volatility adjustment
- **Multi-layer Risk Control**: Portfolio risk, correlation risk, dan market risk
- **Real-time Monitoring**: Monitoring risiko real-time dengan alert otomatis
- **Drawdown Protection**: Perlindungan terhadap drawdown berlebihan
- **Adaptive Stop Loss**: Stop loss yang menyesuaikan dengan volatilitas pasar

### 🤖 Telegram Integration
- **Real-time Signals**: Signal trading real-time dengan analisis lengkap
- **Interactive Charts**: Chart interaktif dengan indikator teknikal
- **Portfolio Tracking**: Tracking portfolio dan performance real-time
- **News Alerts**: Alert berita high-impact dengan analisis sentiment
- **Multi-language Support**: Dukungan 7 bahasa (EN, ID, ES, FR, DE, JA, ZH)
- **Voice Commands**: Perintah suara untuk interaksi yang lebih mudah

### 🔄 N8N Workflow Automation
- **Data Collection**: Otomasi pengumpulan data dari multiple sources
- **Signal Processing**: Pemrosesan signal dengan workflow kompleks
- **Risk Management**: Workflow otomatis untuk manajemen risiko
- **Performance Tracking**: Tracking performance dengan machine learning
- **Alert System**: Sistem alert multi-channel (Telegram, Email, Slack)

### 📈 Performance Monitoring
- **Real-time Metrics**: Metrics performance real-time
- **Advanced Analytics**: Analisis performance dengan 20+ KPI
- **Backtesting Engine**: Engine backtesting dengan data historis
- **Benchmark Comparison**: Perbandingan dengan benchmark market
- **Risk Analytics**: Analisis risiko komprehensif (VaR, Expected Shortfall)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- 8GB RAM minimum (16GB recommended)
- 50GB storage space

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/agi-forex-trading-system.git
cd agi-forex-trading-system
```

### 2. Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 3. Configure API Keys
Edit `.env` file dan isi API keys yang diperlukan:
- OANDA API Key (untuk data forex)
- Telegram Bot Token (untuk notifikasi)
- Alpha Vantage API Key (untuk data fundamental)
- Twitter API Keys (untuk sentiment analysis)

### 4. Start System
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f agi-forex
```

### 5. Access Interfaces
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **N8N Workflows**: http://localhost:5678 (admin/admin123)
- **Kibana Logs**: http://localhost:5601

## 📱 Telegram Bot Setup

### 1. Create Telegram Bot
1. Chat dengan @BotFather di Telegram
2. Gunakan command `/newbot`
3. Ikuti instruksi untuk membuat bot
4. Copy bot token ke `.env` file

### 2. Setup Chat IDs
```bash
# Get your chat ID
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates"
```

### 3. Bot Commands
- `/start` - Mulai bot
- `/signals` - Dapatkan signal terbaru
- `/portfolio` - Lihat portfolio
- `/analysis` - Analisis pasar
- `/performance` - Statistik performance
- `/news` - Berita forex terbaru
- `/settings` - Pengaturan bot

## 🔧 Configuration

### System Configuration
File konfigurasi utama: `config/config.yaml`

```yaml
system:
  name: "AGI Forex Trading System"
  version: "2.0.0"
  environment: "production"

agi_brain:
  models:
    neural_ensemble:
      hidden_sizes: [256, 512, 1024, 512, 256]
      learning_rate: 0.001
    
risk_management:
  max_risk_per_trade: 0.02  # 2%
  max_drawdown: 0.15        # 15%
  position_sizing: "kelly_criterion"
```

### Trading Pairs
Edit `currency_pairs` di config untuk menambah/mengurangi pairs:
```yaml
currency_pairs:
  major:
    - "EURUSD"
    - "GBPUSD"
    - "USDJPY"
  minor:
    - "EURGBP"
    - "EURJPY"
```

## 📊 Monitoring & Analytics

### Grafana Dashboards
- **Trading Performance**: Metrics trading real-time
- **System Health**: Status sistem dan komponen
- **Market Analysis**: Analisis pasar dan volatilitas
- **Risk Monitoring**: Monitoring risiko portfolio

### Key Metrics
- **Win Rate**: Persentase trade yang profit
- **Profit Factor**: Ratio gross profit vs gross loss
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Drawdown maksimum
- **Calmar Ratio**: Annual return / max drawdown

## 🔄 N8N Workflows

### Available Workflows
1. **Market Data Collection**: Pengumpulan data real-time
2. **Signal Generation**: Generasi signal trading
3. **News Processing**: Pemrosesan berita dan sentiment
4. **Risk Monitoring**: Monitoring risiko real-time
5. **Performance Tracking**: Tracking performance otomatis

### Custom Workflows
Buat workflow custom di N8N interface:
1. Akses http://localhost:5678
2. Login dengan admin/admin123
3. Import workflow dari `n8n_workflows/`
4. Customize sesuai kebutuhan

## 🛡️ Security

### API Security
- JWT authentication untuk API
- Rate limiting per endpoint
- IP whitelisting (optional)
- API key rotation otomatis

### Data Protection
- Enkripsi data sensitif (AES-256)
- Secure credential storage
- Audit logging
- GDPR compliance

### Network Security
- SSL/TLS encryption
- Firewall rules
- VPN access (recommended)
- Regular security updates

## 🔧 Development

### Local Development
```bash
# Clone repository
git clone https://github.com/yourusername/agi-forex-trading-system.git
cd agi-forex-trading-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run in development mode
python main.py --debug
```

### Testing
```bash
# Run unit tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run integration tests
python -m pytest tests/integration/ -v
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 📈 Performance Optimization

### System Requirements
- **Minimum**: 4 CPU cores, 8GB RAM, 50GB storage
- **Recommended**: 8 CPU cores, 16GB RAM, 100GB SSD
- **Production**: 16 CPU cores, 32GB RAM, 200GB NVMe SSD

### Optimization Tips
1. **Database Tuning**: Optimize PostgreSQL untuk time-series data
2. **Redis Caching**: Gunakan Redis untuk caching data real-time
3. **Model Optimization**: Optimize model parameters untuk speed vs accuracy
4. **Data Pipeline**: Optimize data collection dan processing pipeline

## 🔄 Backup & Recovery

### Automated Backups
- **Database**: Daily PostgreSQL dumps
- **Models**: Weekly model checkpoints
- **Configuration**: Daily config backups
- **Logs**: Compressed log archives

### Recovery Procedures
```bash
# Restore database
docker exec -i agi-postgres psql -U forex_user -d forex_agi_db < backup.sql

# Restore models
cp model_backup.pkl models/

# Restart system
docker-compose restart
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Bot tidak merespon
```bash
# Check bot status
docker-compose logs agi-forex | grep telegram

# Restart bot
docker-compose restart agi-forex
```

#### 2. Data collection error
```bash
# Check API keys
docker-compose exec agi-forex python -c "import os; print(os.getenv('OANDA_API_KEY'))"

# Check network connectivity
docker-compose exec agi-forex curl -I https://api.oanda.com
```

#### 3. High memory usage
```bash
# Check memory usage
docker stats

# Optimize model parameters
# Edit config/config.yaml - reduce batch_size, hidden_sizes
```

### Log Analysis
```bash
# View real-time logs
docker-compose logs -f agi-forex

# Search for errors
docker-compose logs agi-forex | grep ERROR

# View specific component logs
docker-compose logs n8n
docker-compose logs postgres
```

## 📚 API Documentation

### Authentication
```bash
# Get API token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'
```

### Endpoints
- `GET /signals/latest` - Get latest signals
- `POST /signals/generate` - Generate signal for pair
- `GET /performance` - Get performance metrics
- `GET /portfolio` - Get portfolio status
- `POST /trading/enable` - Enable trading
- `POST /trading/disable` - Disable trading

### WebSocket
```javascript
// Real-time signal updates
const ws = new WebSocket('ws://localhost:8000/ws/signals');
ws.onmessage = (event) => {
  const signal = JSON.parse(event.data);
  console.log('New signal:', signal);
};
```

## 🤝 Contributing

### Development Workflow
1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

### Code Standards
- Follow PEP 8 for Python code
- Add docstrings untuk semua functions
- Write unit tests untuk new features
- Update documentation

### Issue Reporting
Gunakan GitHub Issues untuk:
- Bug reports
- Feature requests
- Performance issues
- Documentation improvements

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OANDA**: Forex data provider
- **Telegram**: Bot platform
- **N8N**: Workflow automation
- **TensorFlow**: Machine learning framework
- **FastAPI**: Web framework
- **Docker**: Containerization platform

## 📞 Support

### Community Support
- **GitHub Issues**: Bug reports dan feature requests
- **Telegram Group**: @AGIForexTrading
- **Discord**: AGI Forex Community

### Professional Support
- **Email**: support@agi-forex.com
- **Documentation**: https://docs.agi-forex.com
- **Training**: https://training.agi-forex.com

---

## 🚀 Roadmap

### Version 2.1 (Q2 2024)
- [ ] Multi-broker support (MT4/MT5, Interactive Brokers)
- [ ] Advanced portfolio optimization
- [ ] Cryptocurrency trading support
- [ ] Mobile app (React Native)

### Version 2.2 (Q3 2024)
- [ ] Options trading strategies
- [ ] Social trading features
- [ ] Advanced backtesting engine
- [ ] Cloud deployment (AWS/GCP/Azure)

### Version 3.0 (Q4 2024)
- [ ] Full AGI implementation
- [ ] Multi-asset trading (Stocks, Commodities, Crypto)
- [ ] Advanced risk management
- [ ] Institutional features

---

**⚠️ Disclaimer**: Trading forex melibatkan risiko tinggi dan dapat menyebabkan kerugian. Sistem ini adalah untuk tujuan edukasi dan penelitian. Selalu lakukan due diligence dan konsultasi dengan advisor keuangan sebelum trading dengan uang riil.

**🔒 Security Notice**: Jangan pernah share API keys atau credentials di public repositories. Gunakan environment variables dan secure storage untuk informasi sensitif.

**📊 Performance Note**: Past performance tidak menjamin hasil di masa depan. Selalu test sistem dengan paper trading sebelum menggunakan uang riil.