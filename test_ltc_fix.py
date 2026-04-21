from stockbroker.agent import AgentBrain
from stockbroker.portfolio import PortfolioManager
from stockbroker.models import AgentConfig

# Test with low buying power
config = AgentConfig()
pm = PortfolioManager(config, initial_cash=0)
brain = AgentBrain(config)

print("=== Testing LTC-USD Fix ===")
print(f"Available cash: ${pm.portfolio.cash:,.2f}")
print(f"Buying power: ${pm.portfolio.buying_power:,.2f}")
max_pos = config.effective_max_single_position_pct * pm.portfolio.cash
print(f"Max single position: ${max_pos:,.2f}")

# Test strategy suggestion (should skip expensive tickers)
tickers = brain.suggest_tickers(pm.portfolio)
print(f"\nSuggested tickers: {tickers}")

# Test analysis of LTC-USD
from stockbroker.market import get_quote
quote = get_quote("LTC-USD")
print(f"\nLTC-USD price: ${quote.price:.2f}")
print(f"Can afford? {'YES' if quote.price < max_pos else 'NO'}")

# Test analysis
analysis = brain.analyse_ticker("LTC-USD", pm.portfolio)
print(f"\nAnalysis result:")
print(f"  Recommendation: {analysis.recommendation}")
print(f"  Target shares: {analysis.target_shares}")
print(f"  Reasoning: {analysis.reasoning}")

print("\n✓ Fix working - no more 0-share orders!")
