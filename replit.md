# XAUUSD Trading Bot Pro V2

## Overview
This project is an automated XAUUSD trading bot accessible via Telegram. It provides real-time trading signals, automatic position tracking, and trade outcome notifications. The bot offers 24/7 unlimited signals, robust risk management, and a database for performance tracking. It is a private bot with access control, features advanced chart generation with technical indicators, and employs a refined dual-mode strategy (Auto/Manual signals). This strategy incorporates advanced filtering with Twin Range Filter (TRF) and Market Bias CEREBR, along with a Trend-Plus-Pullback approach for enhanced precision and opportunity, aiming to be a professional and informative trading assistant for XAUUSD.

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
The bot's architecture is modular, designed for scalability and maintainability.

**Core Components:**
- **Orchestrator:** Manages bot components.
- **Market Data:** Handles Deriv WebSocket connection, OHLC candle construction, and persistence.
- **Strategy:** Implements a dual-mode signal detection (Auto/Manual) using multiple indicators (Twin Range Filter, Market Bias CEREBR, EMA 50, RSI) with a weighted scoring system for signal generation. Auto mode requires a minimum weighted score and trend strength.
- **Position Tracker:** Monitors real-time trade positions per user.
- **Telegram Bot:** Manages command handling and notifications.
- **Chart Generator:** Creates professional charts with integrated technical indicators.
- **Risk Manager:** Calculates lot sizes, P/L, and enforces per-user risk limits (e.g., fixed SL, dynamic TP, signal cooldown).
- **Database:** SQLite for persistent data with PostgreSQL support, auto-migration, connection pooling, and robust transaction management.
- **User Manager:** Handles user authentication and access control via `AUTHORIZED_USER_IDS` and `ID_USER_PUBLIC`.
- **Resilience:** Implements CircuitBreaker for WebSocket, global rate limiting for Telegram API, retry mechanisms, and advanced WebSocket recovery with exponential backoff.
- **System Health:** Includes port conflict detection, bot instance locking (file-based flock), Sentry integration, comprehensive health checks, and OOM graceful degradation.
- **Thread Safety:** Implements asyncio.Lock for position tracking, signal session management, and atomic command execution to prevent race conditions.

**UI/UX Decisions:**
- Telegram serves as the primary user interface.
- Signal messages are enriched with icons, source labels, and confidence reasons.
- Charts display candlesticks, volume, EMA, RSI, Stochastic, TRF bands, and CEREBR in a multi-panel layout.
- Exit notifications (WIN/LOSE) are text-only, while initial signal activations include photo charts.
- All timestamps are displayed in WIB (Asia/Jakarta) timezone.

**Technical Implementations & Feature Specifications:**
- **Indicators:** EMA (5, 10, 20, 50), RSI (14), Stochastic (K=14, D=3), ATR (14), MACD (12,26,9), Volume, Twin Range Filter, Market Bias CEREBR. Includes robust data handling.
- **Risk Management:** Fixed SL ($1 per trade), dynamic TP (1.45x-2.50x R:R), max spread (5 pips), risk per trade (0.5%). Features dynamic SL tightening and trailing stop activation. Unlimited mode disables signal cooldowns and daily loss limits. Lot size is fixed at 0.01.
- **Access Control:** Private bot with dual-tier access.
- **Commands:** Admin commands (`/riset`, `/status`, `/tasks`, `/analytics`, `/systemhealth`) and User commands (`/start`, `/help`, `/monitor`, `/getsignal`, `/status`, `/riwayat`, `/performa`).
- **Anti-Duplicate Protection:** Employs a two-phase cache pattern (pending/confirmed status, hash-based tracking, thread-safe locking, TTL-backed signal cache with async cleanup) for race-condition-safe signal deduplication and chart cleanup. Incorporates anti-spam features like minimum price movement and cooldowns per signal type, generating signals only on M1 candle close.
- **Candle Data Persistence:** Stores M1 and M5 candles in the database for instant bot readiness on restart.
- **Chart Generation:** Uses `mplfinance` and `matplotlib` for multi-panel charts, configured for headless operation, with configurable timeouts and aggressive cleanup.
- **Multi-User Support:** Implements per-user position tracking and risk management.
- **Deployment:** Optimized for Koyeb and Replit, featuring an HTTP server for health checks and webhooks, and `FREE_TIER_MODE` for resource efficiency.
- **Performance Optimization:** Unlimited mode features no global signal cooldown, no tick throttling, early exit for position monitoring, optimized Telegram timeout handling, and fast text-only exit notifications.
- **Logging & Error Handling:** Rate-limited logging, log rotation, type-safe indicator computations, and comprehensive exception handling.
- **Task Management:** Centralized task registry with shielded cancellation for graceful shutdown, background task callbacks, and stale task detection.

## External Dependencies
- **Deriv WebSocket API:** For real-time XAUUSD market data.
- **Telegram Bot API (`python-telegram-bot`):** For all Telegram interactions.
- **SQLAlchemy:** ORM for database interactions (SQLite, PostgreSQL).
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **mplfinance & Matplotlib:** For generating financial charts.
- **pytz:** For timezone handling.
- **aiohttp:** For asynchronous HTTP server and client operations.
- **python-dotenv:** For managing environment variables.
- **Sentry:** For advanced error tracking and monitoring.

## Recent Changes
- **2025-11-27:** SCALPING MODE AGGRESSIVE - Major relaxation of signal filters:
  - Threshold score AUTO diturunkan dari 60% ke 30%
  - Threshold score MANUAL diturunkan dari 40% ke 20%
  - ADX Filter DISABLED - Tidak lagi memblokir market sideways/ranging
  - Session Filter DISABLED - Trading 24/7 diizinkan semua sesi
  - EMA Slope Filter DISABLED - Tidak lagi memblokir sinyal
  - RSI Level Filter DISABLED - Tidak lagi memblokir di overbought/oversold
  - Volume threshold diturunkan dari 80% ke 30%
  - RSI entry range diperluas dari [25-75] ke [15-85]
  - Stochastic levels diperluas dari [15-85] ke [10-90]
  - SIGNAL_MINIMUM_PRICE_MOVEMENT diturunkan dari 0.50 ke 0.05
  - TICK_COOLDOWN_FOR_SAME_SIGNAL diset ke 0 untuk unlimited sinyal
  - Max spread ditingkatkan dari 20 ke 50 pips
  - Logika mandatory filters diubah dari AND ke OR untuk lebih banyak sinyal
  - Semua filter sekarang informational only, tidak blocking
- **2025-11-27:** Added 5 optimization filters to reduce false signals in sideways markets:
  - ADX Filter: Minimum trend strength (ADX >= 20) to skip ranging markets
  - RSI Level Filter: Prevents BUY at overbought (>70) and SELL at oversold (<30)
  - Pin Bar Stricter: Increased tail requirement from 60% to 66.67% (2/3)
  - EMA Slope Filter: Ensures EMA is actively trending, not flat
  - Session Filter: Targets London+NY sessions (14:00-23:00 WIB / 07:00-16:00 UTC)
- **2025-11-27:** Updated config.py with new parameters: ADX_PERIOD, ADX_THRESHOLD, RSI_BUY_MAX_LEVEL, RSI_SELL_MIN_LEVEL, EMA_SLOPE_FILTER_ENABLED, SESSION_FILTER_STRICT, etc.
- **2025-11-27:** Fixed undefined `ErrorHandlingError` exception class in `bot/error_handler.py` - removed 5 references to non-existent exception class that was causing LSP errors.