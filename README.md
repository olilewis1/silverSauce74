# AI Stockbroker Agent

An AI-powered stockbroker agent that analyses stocks, manages risk, and makes investment decisions on your behalf — with both paper trading and **real money** execution via Alpaca.

## Features

- **AI-Powered Analysis** — Uses GPT-4o to analyse stocks, assess risk, and make buy/sell/hold decisions
- **Risk Profiles** — Choose from conservative, moderate, or aggressive strategies
- **Asset Class Filter** — Restrict trading to `stocks`, `crypto`, or `both`
- **Investment Caps** — Set a max % of funds to invest, or a hard dollar cap
- **Position Limits** — Automatic per-position sizing based on risk level
- **Paper Trading** — Simulated execution with full portfolio tracking
- **Live Trading** — Real money execution via Alpaca brokerage (commission-free)
- **Live Market Data** — Real-time quotes and historical data via Yahoo Finance
- **Rich CLI** — Beautiful interactive terminal interface
- **Autonomous Daemon Mode** — Hands-off continuous trading with stop-loss, take-profit, and daily caps

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
# For real money trading, also add ALPACA_API_KEY and ALPACA_SECRET_KEY

# 3. Run the agent
python main.py
```

## Commands

| Command     | Description                                           |
|-------------|-------------------------------------------------------|
| `analyse`   | AI analysis of a single stock                         |
| `suggest`   | Ask the AI for stock picks based on your portfolio    |
| `auto`      | Auto-invest: AI picks stocks, analyses, and trades    |
| `daemon`    | **Start autonomous trading mode** (runs continuously) |
| `buy`       | Manually buy shares of a stock                        |
| `sell`      | Manually sell shares of a stock                       |
| `portfolio` | View current holdings and P&L                         |
| `history`   | View all past trades                                  |
| `refresh`   | Update all position prices                            |
| `deposit`   | Add more cash to the portfolio                        |
| `quit`      | Exit the agent                                        |

## Risk Profiles

| Level        | Max Portfolio % | Max Position % | Description                                    |
|--------------|-----------------|----------------|------------------------------------------------|
| Conservative | 40%             | 10%            | Blue-chip, dividend stocks, low volatility     |
| Moderate     | 65%             | 20%            | Mix of growth & value across sectors           |
| Aggressive   | 90%             | 35%            | Growth, small-caps, momentum plays             |

You can override any of these defaults at startup.

## How It Works

1. You deposit cash and configure risk tolerance
2. You choose whether the agent can trade `stocks`, `crypto`, or `both`
3. The agent fetches live market data (price, volume, fundamentals, volatility)
4. GPT-4o analyses the data against your risk profile and portfolio
5. The risk engine validates every trade against your constraints
6. Approved trades are executed (paper or live via Alpaca)

## Live Trading with Alpaca

To trade with **real money**, you need a free [Alpaca](https://alpaca.markets) account.

### Setup

1. Sign up at [alpaca.markets](https://alpaca.markets) (free, no minimum deposit)
2. Go to your dashboard → API Keys → Generate New Key
3. Add your keys to `.env`:

```bash
ALPACA_API_KEY=your-api-key
ALPACA_SECRET_KEY=your-secret-key
ALPACA_MODE=live      # or "paper" to use Alpaca's paper trading
```

4. Fund your Alpaca account (bank transfer, wire, etc.)
5. Run `python main.py` and select **live** mode at startup
6. Choose the asset filter: **stocks**, **crypto**, or **both**

### How money flows

- Your **cash lives in your Alpaca brokerage account** — not in this app
- When the agent buys stock, it submits a **real market order** to Alpaca
- Alpaca executes the order on the stock exchange
- The agent syncs positions and cash from Alpaca after every trade
- You can see your positions in both this CLI and the Alpaca dashboard

### Safety features for live mode

- **Double confirmation** before enabling live trading
- **All risk limits still enforced** (max %, position caps, risk profile)
- **Stop-loss / take-profit** in daemon mode protects against big losses
- **Daily trade caps** prevent runaway trading
- Orders are **market orders with DAY time-in-force** (cancel if not filled same day)

### Paper vs Live via Alpaca

Alpaca also offers their own paper trading environment. Set `ALPACA_MODE=paper` in `.env` to use Alpaca's paper trading with real market data but fake money — a good middle ground before going fully live.

## Autonomous Daemon Mode

Type `daemon` at the command prompt to enter always-on autonomous trading.
The daemon will ask you to configure:

| Setting               | Default | Description                                           |
|-----------------------|---------|-------------------------------------------------------|
| Check interval        | 30 min  | How often the agent wakes up to analyse and trade     |
| Max trades/day        | 5       | Hard cap on total trades in a calendar day            |
| Max trades/cycle      | 2       | Cap per analysis cycle                                |
| Watchlist             | AI      | Specific tickers, or let the AI suggest each cycle    |
| Market hours only     | Yes     | Skip cycles outside US market hours (9:30–16:00 ET)   |
| Stop-loss %           | 8%      | Auto-sell if a position drops this much               |
| Take-profit %         | 25%     | Auto-sell if a position gains this much               |
| Cooldown after trade  | 60 min  | Wait before trading the same ticker again             |

### How it runs

Each cycle the daemon:
1. Checks if the market is open (if configured)
2. Refreshes all position prices
3. Checks stop-loss / take-profit on existing positions
4. Analyses watchlist tickers (or asks AI for picks)
5. Executes approved trades within daily/cycle limits
6. Sleeps until the next cycle

All activity is logged to both the terminal and `daemon.log`.
Press **Ctrl+C** at any time to stop — the daemon finishes its current cycle and prints a session summary.

If you choose **crypto-only**, the daemon will not be limited by stock market hours, since crypto trades 24/7.

## Project Structure

```
stockbroker/
├── __init__.py      # Package init
├── models.py        # Data models (Portfolio, Trade, Position, RiskProfile)
├── market.py        # Yahoo Finance market data
├── risk.py          # Risk engine & constraint enforcement
├── portfolio.py     # Portfolio management (paper + live)
├── broker.py        # Alpaca broker integration (real money)
├── agent.py         # AI brain (OpenAI integration)
├── daemon.py        # Autonomous trading daemon
└── cli.py           # Rich interactive CLI
```

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4o)
- Internet connection (for market data)
- **For live trading:** Free Alpaca account + API keys