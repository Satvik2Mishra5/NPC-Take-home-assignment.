import logging
from decimal import Decimal
from typing import Dict, List

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

class SmartMarketMaker(ScriptStrategyBase):
    """
    Custom PMM with volatility-based spread, RSI trend filter, and inventory-based order sizing.
    """
    base_spread = 0.001
    min_spread = 0.0005
    max_spread = 0.005
    order_refresh_time = 15
    base_order_amount = 0.01
    create_timestamp = 0
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice

    # Candles config
    candle_exchange = "binance"
    candles_interval = "1m"
    candles_length = 30
    max_records = 1000

    candles = CandlesFactory.get_candle(CandlesConfig(connector=candle_exchange,
                                                      trading_pair=trading_pair,
                                                      interval=candles_interval,
                                                      max_records=max_records))

    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()

    def on_stop(self):
        self.candles.stop()

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp and self.ready_to_trade:
            self.cancel_all_orders()

            # Create and adjust proposals
            proposal = self.create_proposal()
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)

            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.current_timestamp + self.order_refresh_time

    def get_candles_with_indicators(self):
        candles_df = self.candles.candles_df.copy()
        candles_df.ta.rsi(length=14, append=True)
        candles_df['volatility'] = candles_df['close'].rolling(10).std()
        return candles_df

    def dynamic_spread(self, volatility: float) -> float:
        return float(min(max(self.base_spread + volatility * 0.5, self.min_spread), self.max_spread))

    def create_proposal(self) -> List[OrderCandidate]:
        ref_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        base_asset, quote_asset = self.trading_pair.split("-")
        balances = self.connectors[self.exchange].get_balance(base_asset)

        # Fetch candles and indicators
        candles_df = self.get_candles_with_indicators()
        latest = candles_df.iloc[-1]
        rsi = latest["RSI_14"]
        volatility = latest["volatility"]

        spread = self.dynamic_spread(volatility)
        buy_price = ref_price * Decimal(1 - spread)
        sell_price = ref_price * Decimal(1 + spread)

        # Inventory-based order size adjustment
        quote_balance = self.connectors[self.exchange].get_balance(quote_asset)
        base_balance = self.connectors[self.exchange].get_balance(base_asset)
        total_value = quote_balance + base_balance * ref_price
        inventory_ratio = (base_balance * ref_price) / total_value if total_value > 0 else 0.5

        # Reduce order size if too skewed
        if inventory_ratio > 0.8 or inventory_ratio < 0.2:
            order_amount = Decimal(self.base_order_amount) * Decimal(0.5)
        else:
            order_amount = Decimal(self.base_order_amount)

        # RSI trend filter: Don't place if extreme trends
        if rsi > 75 or rsi < 25:
            self.log_with_clock(logging.INFO, f"Trend too strong (RSI={rsi:.2f}), skipping order.")
            return []

        buy_order = OrderCandidate(
            trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
            order_side=TradeType.BUY, amount=order_amount, price=buy_price)

        sell_order = OrderCandidate(
            trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
            order_side=TradeType.SELL, amount=order_amount, price=sell_price)

        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        return self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)

    def place_orders(self, proposal: List[OrderCandidate]):
        for order in proposal:
            self.place_order(self.exchange, order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name, order.trading_pair, order.amount, order.order_type, order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name, order.trading_pair, order.amount, order.order_type, order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = f"{event.trade_type.name} {round(event.amount, 4)} {event.trading_pair} at {round(event.price, 2)}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Bot not ready"
        lines = ["", "  Balances:"]
        balance_df = self.get_balance_df()
        lines += ["    " + line for line in balance_df.to_string(index=False).split("\n")]

        try:
            df = self.active_orders_df()
            lines += ["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")]
        except ValueError:
            lines += ["", "  No active orders"]

        lines.append("\n--- Candles & Indicators ---\n")
        candles_df = self.get_candles_with_indicators()
        lines += ["    " + line for line in candles_df.tail(5).iloc[::-1].to_string(index=False).split("\n")]

        return "\n".join(lines)
