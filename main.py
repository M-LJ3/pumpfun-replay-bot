"""Pump.fun replay sprint bot: buy a coin small and cheap, then give it
time to run.

Strategy, designed from the dev dataset (1,222 universe coins, 24h window):
every coin pumps once and then retraces to its creation price by the end
of the window, and a plain momentum buy loses because fills land about
39% above the creation price while the two-slot execution lag and the
~4.6% round-trip fees eat the rest. Two things move the result: (1) only
buy a coin whose live price is still at most 1.07x its creation price --
a coin that has already pumped hard has little upside left and a far
worse fill, while a coin that barely moved on its first trade still has
room; (2) hold the position for a long time (up to 50,000 slots, about
three hours of chain time) and take profit at 10% above the fill cost, so
the few coins that do run keep running and the losers are simply held
until they flatten near the creation price, losing only the small
retrace plus fees. No stop-loss (it cut coins that later recovered), no
adds (buying more of a pumping coin raised the blended cost), no
re-entry, no shorts, no leverage, no randomness, no clock, no network:
every decision is a pure function of the replay state, so two runs
produce identical action logs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bot  # noqa: E402

SOL = makefun_bot.LAMPORTS_PER_SOL

#: First stake per coin (0.01 SOL): the round-trip minimum, bought the first
#: time the size cap allows it (real quote reserves >= 1 SOL).
FIRST_STAKE_LAMPORTS = SOL // 100
#: Only buy a coin whose live price is at most this multiple of its creation
#: price (107/100): a coin that has already pumped hard by decision time
#: has little upside left and a far worse fill.
MAX_ENTRY_NUM = 107
MAX_ENTRY_DEN = 100
#: Only consider coins this young for new entries: 1800 slots (~6 minutes).
BUY_MAX_AGE_SLOTS = 1800
#: Take profit: live price at least 10% above the fill cost (11/10).
TP_NUM = 11
TP_DEN = 10
#: Stop loss is disabled (SL_NUM = 0, so the stop-loss expression is always
#: false): cutting losers early cost more than it saved, because coins that
#: dip below cost often drift back up before the hold ends.
SL_NUM = 0
SL_DEN = 10
#: Longest hold from the first fill: sell after 50,000 slots (~3 hours of
#: chain time) if the take profit has not fired.
MAX_HOLD_SLOTS = 50000
BUY_SLIPPAGE_BPS = 1000
SELL_SLIPPAGE_BPS = 10000

#: mint -> (creation virtual quote, creation virtual token, creation slot)
created = {}
#: mint -> slot of the first fill (for the maximum hold).
first_fill_slot = {}
#: mints with a buy decided and not yet reported.
pending_buy = set()
#: mints with a sell decided and not yet reported.
pending_sell = set()
#: mints closed once already (never re-entered).
closed = set()
#: The bot's own action log: one entry per batch, {"slot": slot,
#: "orders": [order, ...]}. Kept in memory only; the self-check in
#: audit_action_log re-reads it after the replay.
ACTION_LOG = []


def audit_action_log(log):
    """Re-read the bot's own action log and confirm the competition
    guardrails held: at least 20 distinct coins with a completed round
    trip, never more than 5 open positions at once, and no coin re-entered
    after it was closed. Prints a one-line summary to stderr (never
    stdout, so the replay protocol stream stays clean) and returns True
    when every guardrail passes."""
    bought = {}
    sold = {}
    open_now = set()
    max_open = 0
    reentered = set()
    for entry in log:
        for order in entry["orders"]:
            mint = order["mint"]
            if order["side"] == "buy":
                if mint in sold and mint not in open_now:
                    reentered.add(mint)
                bought[mint] = bought.get(mint, 0) + 1
                open_now.add(mint)
            else:
                sold[mint] = sold.get(mint, 0) + 1
                open_now.discard(mint)
        if len(open_now) > max_open:
            max_open = len(open_now)
    round_trips = sum(1 for mint in bought if mint in sold)
    ok = round_trips >= 20 and max_open <= 5 and not reentered
    print(
        "audit_action_log: distinct_bought=%d round_trips=%d max_open=%d "
        "reentered=%d -> %s"
        % (len(bought), round_trips, max_open, len(reentered), "PASS" if ok else "FAIL"),
        file=sys.stderr,
    )
    return ok


def on_batch(state, batch):
    for fill in batch["fills"]:
        mint = fill["mint"]
        if fill["side"] == "buy":
            pending_buy.discard(mint)
        if fill["side"] == "sell":
            pending_sell.discard(mint)
    for event in batch["events"]:
        if event["event"] == "CreateEvent":
            mint = event["mint"]
            if mint not in created:
                created[mint] = (
                    int(event["virtual_quote_reserves"]),
                    int(event["virtual_token_reserves"]),
                    event["slot"],
                )
    for mint in list(first_fill_slot):
        if mint not in state.positions:
            closed.add(mint)
            pending_sell.discard(mint)
            first_fill_slot.pop(mint, None)
    for mint in state.positions:
        if mint not in first_fill_slot:
            first_fill_slot[mint] = state.slot
    orders = []
    caps = state.hello["caps"]
    tx = state.tx_fee_lamports()
    free = state.free_cash_lamports()
    slots = caps["max_open_positions"] - state.committed_position_count()
    for mint in sorted(state.positions):
        if mint in pending_sell:
            continue
        holding = state.positions[mint]
        curve = state.curves.get(mint)
        if curve is None or curve.complete:
            continue
        vq = curve.virtual_quote_reserves
        vt = curve.virtual_token_reserves
        tokens = holding.token_amount
        cost = holding.cost_lamports
        if tokens <= 0 or cost <= 0:
            continue
        take_profit = vq * tokens * TP_DEN >= TP_NUM * cost * vt
        stop_loss = vq * tokens * SL_DEN <= SL_NUM * cost * vt
        too_old = state.slot - first_fill_slot.get(mint, state.slot) >= MAX_HOLD_SLOTS
        if take_profit or stop_loss or too_old:
            orders.append(
                {
                    "side": "sell",
                    "mint": mint,
                    "token_amount": tokens,
                    "max_slippage_bps": SELL_SLIPPAGE_BPS,
                }
            )
            pending_sell.add(mint)
            slots += 1
            continue
    if len(orders) < caps["max_orders_per_batch"]:
        candidates = []
        for mint, (vq0, vt0, create_slot) in created.items():
            if mint in closed or mint in pending_buy or mint in state.positions:
                continue
            if state.slot - create_slot > BUY_MAX_AGE_SLOTS:
                continue
            curve = state.curves.get(mint)
            if curve is None or curve.complete:
                continue
            if (
                curve.virtual_quote_reserves * vt0 * MAX_ENTRY_DEN
                > MAX_ENTRY_NUM * vq0 * curve.virtual_token_reserves
            ):
                continue
            cap = state.max_buy_lamports(mint)
            if cap < FIRST_STAKE_LAMPORTS:
                continue
            candidates.append((create_slot, mint))
        candidates.sort()
        for _create_slot, mint in candidates:
            if slots <= 0 or len(orders) >= caps["max_orders_per_batch"]:
                break
            if free < FIRST_STAKE_LAMPORTS + tx:
                continue
            orders.append(
                {
                    "side": "buy",
                    "mint": mint,
                    "quote_lamports": FIRST_STAKE_LAMPORTS,
                    "max_slippage_bps": BUY_SLIPPAGE_BPS,
                }
            )
            pending_buy.add(mint)
            free -= FIRST_STAKE_LAMPORTS + tx
            slots -= 1
    ACTION_LOG.append({"slot": state.slot, "orders": orders})
    return orders


def main():
    makefun_bot.run(on_batch)
    audit_action_log(ACTION_LOG)


if __name__ == "__main__":
    main()
