# XAUUSD Trading Bot Pro V2

## Overview
This project is an automated Telegram-based trading bot for XAUUSD, designed to provide real-time signals, automatic position tracking, and trade outcome notifications. It offers 24/7 unlimited signals, robust risk management, and performance tracking via a database. Key features include advanced chart generation with technical indicators, a refined dual-mode (Auto/Manual) trading strategy utilizing advanced filtering with Twin Range Filter (TRF) and Market Bias CEREBR, and a Trend-Plus-Pullback approach for enhanced precision. The bot aims to be a professional, informative, and accessible trading assistant for XAUUSD, with a focus on private access control.

## User Preferences
- Bahasa komunikasi: **Bahasa Indonesia** (100% tidak ada bahasa Inggris)
- Data source: **Deriv WebSocket** (gratis, tanpa API key)
- Trading pair: **XAUUSD** (Gold)
- Notifikasi: **Telegram** dengan foto chart + indikator
- Tracking: **Real-time** sampai TP/SL
- Mode: **24/7 unlimited** untuk user terdaftar
- Akurasi: Strategi multi-indicator dengan validasi ketat
- Chart: Menampilkan indikator EMA, RSI, Stochastic (tidak polos)
- Akses Bot: **Privat** - hanya untuk user yang terdaftar di AUTHORIZED_USER_IDS atau ID_USER_PUBLIC

## System Architecture
The bot features a modular architecture for scalability and maintainability.

**Core Components & System Design:**
- **Orchestrator:** Manages bot operations.
- **Market Data:** Handles Deriv WebSocket connection, OHLC candle construction, and persistence.
- **Strategy:** Implements dual-mode signal detection (Auto/Manual) using indicators like Twin Range Filter, Market Bias CEREBR, EMA 50, and RSI, with a weighted scoring system.
- **Position Tracker:** Monitors real-time trade positions per user.
- **Telegram Bot:** Manages command handling and notifications.
- **Chart Generator:** Creates professional charts with integrated technical indicators.
- **Risk Manager:** Calculates lot sizes, P/L, and enforces per-user risk limits (fixed SL, dynamic TP, signal cooldown).
- **Database:** SQLite (with PostgreSQL support) for persistent data, featuring auto-migration, connection pooling, and robust transaction management.
- **User Manager:** Handles user authentication and access control.
- **Resilience:** Incorporates CircuitBreaker for WebSocket, global rate limiting for Telegram API, retry mechanisms, and advanced WebSocket recovery.
- **System Health:** Includes port conflict detection, bot instance locking, Sentry integration, health checks, and OOM graceful degradation.
- **Thread Safety:** Utilizes `asyncio.Lock` for position tracking, signal session management, and atomic command execution.

**UI/UX Decisions:**
- Telegram serves as the primary user interface.
- Signal messages are enriched with icons, source labels, and confidence reasons.
- Charts display candlesticks, volume, EMA, RSI, Stochastic, TRF bands, and CEREBR in a multi-panel layout.
- Exit notifications (WIN/LOSE) are text-only, while initial signal activations include photo charts.
- All timestamps are displayed in WIB (Asia/Jakarta) timezone.

**Technical Implementations & Feature Specifications:**
- **Indicators:** EMA (5, 10, 20, 50), RSI (14), Stochastic (K=14, D=3), ATR (14), MACD (12,26,9), Volume, Twin Range Filter, Market Bias CEREBR.
- **Risk Management:** Fixed SL ($1 per trade), dynamic TP (1.45x-2.50x R:R), max spread (5 pips), risk per trade (0.5%). Includes dynamic SL tightening and trailing stop activation. Lot size is fixed at 0.01.
- **Access Control:** Private bot with dual-tier access.
- **Commands:** Admin commands (`/riset`, `/status`, `/tasks`, `/analytics`, `/systemhealth`) and User commands (`/start`, `/help`, `/monitor`, `/getsignal`, `/status`, `/riwayat`, `/performa`).
- **Anti-Duplicate Protection:** Employs a two-phase cache pattern (pending/confirmed status, hash-based tracking, thread-safe locking, TTL-backed signal cache with async cleanup) for race-condition-safe signal deduplication and anti-spam.
- **Candle Data Persistence:** Stores M1 and M5 candles in the database.
- **Chart Generation:** Uses `mplfinance` and `matplotlib` for multi-panel charts, configured for headless operation.
- **Multi-User Support:** Implements per-user position tracking and risk management.
- **Deployment:** Optimized for Koyeb and Replit, featuring an HTTP server for health checks and webhooks, and `FREE_TIER_MODE` for resource efficiency.
- **Performance Optimization:** Unlimited mode features no global signal cooldown, no tick throttling, early exit for position monitoring, optimized Telegram timeout handling, and fast text-only exit notifications.
- **Logging & Error Handling:** Rate-limited logging, log rotation, type-safe indicator computations, and comprehensive exception handling.
- **Task Management:** Centralized task registry with shielded cancellation for graceful shutdown and background task callbacks.

## External Dependencies
- **Deriv WebSocket API:** For real-time XAUUSD market data.
- **Telegram Bot API (`python-telegram-bot`):** For all Telegram interactions.
- **SQLAlchemy:** ORM for database interactions.
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **mplfinance & Matplotlib:** For generating financial charts.
- **pytz:** For timezone handling.
- **aiohttp:** For asynchronous HTTP server and client operations.
- **python-dotenv:** For managing environment variables.
- **Sentry:** For advanced error tracking and monitoring.
```