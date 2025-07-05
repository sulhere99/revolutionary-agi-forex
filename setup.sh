#!/bin/bash

# AGI Forex Trading System - Setup Script
# =======================================
# Script untuk setup dan instalasi sistem AGI Forex Trading

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        exit 1
    fi
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    log_success "Operating system: $OS"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 ]] && [[ $PYTHON_MINOR -ge 11 ]]; then
            log_success "Python version: $PYTHON_VERSION"
        else
            log_error "Python 3.11+ required, found: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python 3 not found"
        exit 1
    fi
    
    # Check available memory
    if [[ "$OS" == "linux" ]]; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    elif [[ "$OS" == "macos" ]]; then
        MEMORY_BYTES=$(sysctl -n hw.memsize)
        MEMORY_GB=$((MEMORY_BYTES / 1024 / 1024 / 1024))
    fi
    
    if [[ $MEMORY_GB -lt 8 ]]; then
        log_warning "Recommended minimum 8GB RAM, found: ${MEMORY_GB}GB"
    else
        log_success "Available memory: ${MEMORY_GB}GB"
    fi
    
    # Check available disk space
    DISK_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $DISK_SPACE -lt 50 ]]; then
        log_warning "Recommended minimum 50GB disk space, available: ${DISK_SPACE}GB"
    else
        log_success "Available disk space: ${DISK_SPACE}GB"
    fi
}

# Install system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        # Detect Linux distribution
        if command -v apt-get &> /dev/null; then
            # Debian/Ubuntu
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                curl \
                git \
                wget \
                unzip \
                libpq-dev \
                libffi-dev \
                libssl-dev \
                pkg-config \
                gcc \
                g++ \
                make \
                python3-dev \
                python3-pip \
                python3-venv
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                curl \
                git \
                wget \
                unzip \
                postgresql-devel \
                libffi-devel \
                openssl-devel \
                pkgconfig \
                python3-devel \
                python3-pip
        elif command -v pacman &> /dev/null; then
            # Arch Linux
            sudo pacman -S --noconfirm \
                base-devel \
                curl \
                git \
                wget \
                unzip \
                postgresql-libs \
                libffi \
                openssl \
                pkgconf \
                python \
                python-pip
        else
            log_error "Unsupported Linux distribution"
            exit 1
        fi
    elif [[ "$OS" == "macos" ]]; then
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            log_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install dependencies with Homebrew
        brew install \
            curl \
            git \
            wget \
            postgresql \
            libffi \
            openssl \
            pkg-config \
            python@3.11
    fi
    
    log_success "System dependencies installed"
}

# Install TA-Lib
install_talib() {
    log_info "Installing TA-Lib..."
    
    if [[ "$OS" == "linux" ]]; then
        # Download and compile TA-Lib
        cd /tmp
        wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
        tar -xzf ta-lib-0.4.0-src.tar.gz
        cd ta-lib/
        ./configure --prefix=/usr/local
        make
        sudo make install
        cd -
        rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
    elif [[ "$OS" == "macos" ]]; then
        brew install ta-lib
    fi
    
    log_success "TA-Lib installed"
}

# Install Docker
install_docker() {
    log_info "Checking Docker installation..."
    
    if command -v docker &> /dev/null; then
        log_success "Docker already installed"
        return
    fi
    
    log_info "Installing Docker..."
    
    if [[ "$OS" == "linux" ]]; then
        # Install Docker on Linux
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        
        # Install Docker Compose
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        
    elif [[ "$OS" == "macos" ]]; then
        log_info "Please install Docker Desktop for Mac from: https://docs.docker.com/desktop/mac/install/"
        log_warning "After installing Docker Desktop, please restart this script"
        exit 1
    fi
    
    log_success "Docker installed"
    log_warning "Please log out and log back in for Docker group changes to take effect"
}

# Setup Python virtual environment
setup_python_environment() {
    log_info "Setting up Python virtual environment..."
    
    # Create virtual environment
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install wheel and setuptools
    pip install wheel setuptools
    
    log_success "Python virtual environment created"
}

# Install Python dependencies
install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    pip install -r requirements.txt
    
    log_success "Python dependencies installed"
}

# Setup configuration
setup_configuration() {
    log_info "Setting up configuration..."
    
    # Copy environment template
    if [[ ! -f .env ]]; then
        cp .env.example .env
        log_success "Environment file created from template"
        log_warning "Please edit .env file with your API keys and configuration"
    else
        log_info "Environment file already exists"
    fi
    
    # Create necessary directories
    mkdir -p logs data backups config/secrets
    
    # Set permissions
    chmod 700 config/secrets
    
    log_success "Configuration setup completed"
}

# Setup database
setup_database() {
    log_info "Setting up database..."
    
    # Check if PostgreSQL is running
    if command -v pg_isready &> /dev/null; then
        if pg_isready -q; then
            log_success "PostgreSQL is running"
        else
            log_warning "PostgreSQL is not running. Please start PostgreSQL service"
        fi
    else
        log_info "PostgreSQL not found locally. Will use Docker container"
    fi
    
    # Create database initialization scripts
    mkdir -p database/init
    
    cat > database/init/01-create-database.sql << 'EOF'
-- Create database and user for AGI Forex Trading System
CREATE DATABASE forex_agi_db;
CREATE DATABASE n8n_db;

CREATE USER forex_user WITH PASSWORD 'your_password_here';
CREATE USER n8n_user WITH PASSWORD 'your_password_here';

GRANT ALL PRIVILEGES ON DATABASE forex_agi_db TO forex_user;
GRANT ALL PRIVILEGES ON DATABASE n8n_db TO n8n_user;

-- Create extensions
\c forex_agi_db;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

\c n8n_db;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
EOF
    
    log_success "Database setup scripts created"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create monitoring directories
    mkdir -p monitoring/{prometheus,grafana/{dashboards,provisioning/{dashboards,datasources}},logstash/{config,pipeline}}
    
    # Create Prometheus configuration
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'agi-forex'
    static_configs:
      - targets: ['agi-forex:9090']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF
    
    # Create Grafana datasource configuration
    cat > monitoring/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    
    # Create Grafana dashboard configuration
    cat > monitoring/grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF
    
    log_success "Monitoring setup completed"
}

# Setup SSL certificates
setup_ssl() {
    log_info "Setting up SSL certificates..."
    
    mkdir -p ssl
    
    # Generate self-signed certificate for development
    if [[ ! -f ssl/cert.pem ]] || [[ ! -f ssl/key.pem ]]; then
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        log_success "Self-signed SSL certificate generated"
        log_warning "For production, please use proper SSL certificates"
    else
        log_info "SSL certificates already exist"
    fi
}

# Setup Nginx configuration
setup_nginx() {
    log_info "Setting up Nginx configuration..."
    
    mkdir -p nginx/conf.d
    
    # Create main Nginx configuration
    cat > nginx/nginx.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    gzip on;
    gzip_vary on;
    gzip_min_length 10240;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/x-javascript
        application/xml+rss
        application/json;

    include /etc/nginx/conf.d/*.conf;
}
EOF
    
    # Create AGI Forex site configuration
    cat > nginx/conf.d/agi-forex.conf << 'EOF'
upstream agi_forex_backend {
    server agi-forex:8000;
}

upstream grafana_backend {
    server grafana:3000;
}

upstream n8n_backend {
    server n8n:5678;
}

server {
    listen 80;
    server_name localhost;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name localhost;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # API endpoints
    location /api/ {
        proxy_pass http://agi_forex_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check
    location /health {
        proxy_pass http://agi_forex_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Grafana
    location /grafana/ {
        proxy_pass http://grafana_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # N8N
    location /n8n/ {
        proxy_pass http://n8n_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "upgrade";
        proxy_set_header Upgrade $http_upgrade;
    }

    # Default location
    location / {
        proxy_pass http://agi_forex_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF
    
    log_success "Nginx configuration created"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run configuration test
    python run.py --check-config
    
    # Run dependency test
    python run.py --check-dependencies
    
    log_success "Tests completed"
}

# Main setup function
main() {
    echo "🤖 AGI Forex Trading System - Setup Script"
    echo "==========================================="
    echo ""
    
    # Check if running as root
    check_root
    
    # Check system requirements
    check_system_requirements
    
    # Ask for confirmation
    echo ""
    read -p "Do you want to proceed with the installation? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled"
        exit 0
    fi
    
    # Installation steps
    install_system_dependencies
    install_talib
    install_docker
    setup_python_environment
    install_python_dependencies
    setup_configuration
    setup_database
    setup_monitoring
    setup_ssl
    setup_nginx
    
    # Run tests
    run_tests
    
    echo ""
    log_success "🎉 AGI Forex Trading System setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your API keys and configuration"
    echo "2. Start the system with: docker-compose up -d"
    echo "3. Access the API documentation at: https://localhost/docs"
    echo "4. Access Grafana dashboard at: https://localhost/grafana"
    echo "5. Access N8N workflows at: https://localhost/n8n"
    echo ""
    echo "For development mode:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Run in development mode: python run.py --mode development"
    echo ""
    log_warning "Please review the configuration and security settings before using in production!"
}

# Run main function
main "$@"