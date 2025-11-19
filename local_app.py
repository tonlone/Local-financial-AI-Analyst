import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Local Value Investor", layout="wide", page_icon="📈")

# --- CSS STYLING ---
st.markdown("""
<style>
    /* Multiplier Box */
    .multiplier-box {
        font-size: 35px; font-weight: bold; text-align: center; padding: 15px; 
        border-radius: 10px; background-color: #ffffff; margin-top: 10px;
        margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* Methodology Box */
    .methodology-box {
        background-color: #262730; padding: 15px; border-radius: 10px;
        border: 1px solid #444; font-size: 14px; margin-top: 20px;
    }
    .final-score-box {
        text-align: center; padding: 20px; border-radius: 15px; 
        background-color: #ffffff; margin-top: 20px; border: 4px solid #ccc;
    }
    /* Compact Metrics for Tab 3 */
    div[data-testid="stMetricValue"] { font-size: 18px !important; }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; color: #888; }
    
    div[data-testid="stForm"] button[kind="primary"] {
        background-color: #FF4B4B; color: white; border: none;
        font-weight: bold; font-size: 16px; padding: 0.5rem 1rem; width: 100%;
    }
    div[data-testid="stForm"] button[kind="primary"]:hover {
        background-color: #FF0000; border-color: #FF0000;
    }
</style>
""", unsafe_allow_html=True)

# --- LOCAL AI CLIENT SETUP ---
try:
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    connection_status = True
except:
    connection_status = False

# --- DATA FUNCTIONS ---

def fmt_num(val, is_pct=False, is_currency=False):
    """
    General formatter for Margins, ratios, etc.
    Note: Profit Margins usually come as decimals (0.15) so we *100.
    """
    if val is None or val == "N/A": return "-"
    if is_pct: return f"{val * 100:.2f}%"
    if is_currency:
        if val > 1e12: return f"{val/1e12:.2f}T"
        if val > 1e9: return f"{val/1e9:.2f}B"
        if val > 1e6: return f"{val/1e6:.2f}M"
    return f"{val:.2f}"

def fmt_dividend(val):
    """
    Specific formatter for Dividend Yield.
    yfinance returns this as the actual percentage number (e.g., 4.94 or 0.02).
    We do NOT multiply by 100.
    """
    if val is None: return "-"
    return f"{val:.2f}%"

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info: return None
        price = info.get('currentPrice', 0)
        hist = stock.history(period="1y")
        if price == 0 and not hist.empty: price = hist['Close'].iloc[-1]
        eps = info.get('forwardEps', info.get('trailingEps', 0))
        pe = price / eps if eps and eps > 0 else 0
        
        return {
            "price": price, "currency": info.get('currency', 'USD'), "pe": pe,
            "name": info.get('longName', ticker), "industry": info.get('industry', 'Unknown'),
            "summary": info.get('longBusinessSummary', ''), "history": hist,
            "raw_info": info 
        }
    except: return None

def calculate_technicals(df):
    if df.empty or len(df) < 200: return None
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
    curr_vol = df['Volume'].iloc[-1]
    vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
    recent_data = df.tail(60)
    support = recent_data['Low'].min()
    resistance = recent_data['High'].max()
    volatility_short = df['Close'].rolling(window=10).std().iloc[-1]
    volatility_long = df['Close'].rolling(window=60).std().iloc[-1]
    is_squeezing = volatility_short < (volatility_long * 0.5)
    current_price = df['Close'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    sma_200 = df['SMA_200'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    trend = "Neutral"
    if current_price > sma_200:
        trend = "Uptrend 🟢" if current_price > sma_50 else "Weak Uptrend 🟡"
    else:
        trend = "Downtrend 🔴"
    return {
        "trend": trend, "rsi": rsi, "support": support, "resistance": resistance,
        "vol_ratio": vol_ratio, "is_squeezing": is_squeezing,
        "sma_50": sma_50, "sma_200": sma_200, "last_price": current_price
    }

def analyze_qualitative(ticker, summary, topic):
    prompt = (
        f"You are a financial analyst. Analyze {ticker} regarding '{topic}'. "
        f"Context: {summary}. "
        f"Give a specific score from 0.0 to 4.0 (use 1 decimal place). "
        f"Provide a 1 sentence reason. "
        f"Strict Format: SCORE|REASON"
    )
    try:
        resp = client.chat.completions.create(
            model="local-model", 
            messages=[
                {"role": "system", "content": "You are a strict financial analyst. Output only the requested format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=800
        )
        raw_content = resp.choices[0].message.content
        clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        return clean_content, False
    except Exception as e:
        return f"0.0|Error: {str(e)}", True

# --- INPUT LOGIC ---
if 'layout_mode' not in st.session_state: st.session_state.layout_mode = 'desktop' 
if 'active_ticker' not in st.session_state: st.session_state.active_ticker = "NVDA"
if 'active_market' not in st.session_state: st.session_state.active_market = "US"

# --- SIDEBAR ---
with st.sidebar:
    st.header("Analysis Tool")
    with st.form(key='desktop_form'):
        st.caption("Select Market")
        d_market = st.selectbox("Market", ["US", "Canada (TSX)", "HK (HKEX)"], label_visibility="collapsed")
        st.caption("Enter Stock Ticker")
        d_ticker = st.text_input("Ticker", value="NVDA", label_visibility="collapsed").upper()
        d_submit = st.form_submit_button("Analyze Stock", type="primary") 
    
    st.markdown("---")
    
    if connection_status:
        try:
            client.models.list()
            st.success("🟢 LM Studio Connected")
        except: st.error("🔴 LM Studio Disconnected")
    else: st.error("🔴 LM Studio Disconnected")

    st.markdown("""
<div class="methodology-box">
<h4 style="margin-top:0; color: #4da6ff;">Methodology:</h4>
<p style="margin-bottom: 5px;"><strong style="color: #4da6ff;">Qualitative Score (0-20)</strong><br>
<span style="color: #aaa; font-size: 12px;">(5 topics x 4 pts)</span></p>
<p style="text-align:center; margin: 5px 0;">✖</p>
<p style="margin-bottom: 5px;"><strong style="color: #4da6ff;">Valuation Multiplier (1-5)</strong><br>
<span style="color: #aaa; font-size: 12px;">(Based on PE Ratio)</span></p>
<hr style="margin: 10px 0; border-color: #555;">
<p style="margin-bottom: 0;"><strong style="color: #4da6ff;">= Final Score (0-100)</strong></p>
</div>
""", unsafe_allow_html=True)

# --- MOBILE SEARCH ---
with st.expander("📱 Tap here for Mobile Search", expanded=False):
    with st.form(key='mobile_form'):
        m_col1, m_col2 = st.columns([1, 1])
        with m_col1: m_market = st.selectbox("Market", ["US", "Canada (TSX)", "HK (HKEX)"], key='m_m')
        with m_col2: m_ticker = st.text_input("Ticker", value="NVDA", key='m_t').upper()
        m_submit = st.form_submit_button("Analyze (Mobile)", type="primary")

run_analysis = False
if d_submit:
    st.session_state.layout_mode = 'desktop'
    st.session_state.active_ticker = d_ticker
    st.session_state.active_market = d_market
    run_analysis = True
elif m_submit:
    st.session_state.layout_mode = 'mobile'
    st.session_state.active_ticker = m_ticker
    st.session_state.active_market = m_market
    run_analysis = True

# --- MAIN EXECUTION ---
if run_analysis:
    raw_t = st.session_state.active_ticker
    mkt = st.session_state.active_market
    final_t = raw_t
    if mkt == "Canada (TSX)" and ".TO" not in raw_t: final_t += ".TO"
    elif mkt == "HK (HKEX)": 
        nums = ''.join(filter(str.isdigit, raw_t))
        final_t = f"{nums.zfill(4)}.HK" if nums else f"{raw_t}.HK"

    with st.spinner(f"Fetching data for {final_t}..."):
        data = get_stock_data(final_t)

    if data:
        st.header(f"{data['name']} ({final_t})")
        st.caption(f"Industry: {data['industry']} | Currency: {data['currency']}")
        
        tab_fund, tab_tech, tab_fin = st.tabs(["💎 Value Analysis", "📈 Technical Analysis", "📊 Financials"])

        # ==========================================
        # TAB 1: FUNDAMENTAL VALUE
        # ==========================================
        with tab_fund:
            topics = ["Unique Product/Moat", "Revenue Growth", "Competitive Advantage", "Profit Stability", "Management"]
            qual_results = []
            total_qual = 0.0 
            prog_bar = st.progress(0)
            status_text = st.empty()
            
            col_q, col_v = st.columns([1.6, 1])
            
            with col_q:
                st.subheader("1. Qualitative Analysis")
                for i, t in enumerate(topics):
                    prog_bar.progress((i)/5)
                    status_text.text(f"AI Analyzing: {t}...")
                    res, is_error = analyze_qualitative(data['name'], data['summary'], t)
                    match = re.search(r'\b([0-3](?:\.\d)?|4(?:\.0)?)\b', res)
                    if match:
                        s_str = match.group(1); s = float(s_str)
                        r = res.replace(s_str, "").replace("|", "").replace("SCORE", "").replace("REASON", "").strip().strip(' :-=\n')
                    else: s, r = 0.0, res 
                    total_qual += s
                    qual_results.append((t, s, r))
                    with st.expander(f"Analyzing: {t}...", expanded=True): st.markdown(f"**Score: {s}/4** — {r}")

                prog_bar.empty(); status_text.empty()

            pe = data['pe']
            if pe <= 0: mult, color_code, pe_text = 1.0, "#FF4500", "❌ Negative Earnings"
            elif pe <= 20: mult, color_code, pe_text = 5.0, "#00C805", "✅ Undervalued (PE < 20)"
            elif pe >= 75: mult, color_code, pe_text = 1.0, "#FF4500", "⚠️ Overvalued (PE > 75)"
            else:
                pct = (pe - 20) / 55; mult = 5.0 - (pct * 4.0)
                if mult >= 4.0: color_code, pe_text = "#00C805", "✅ Fairly Valued"
                elif mult >= 3.0: color_code, pe_text = "#90EE90", "⚖️ Fair Value"
                elif mult >= 2.0: color_code, pe_text = "#FFA500", "⚠️ Slightly Expensive"
                else: color_code, pe_text = "#FF4500", "⚠️ Expensive"

            mult = round(mult, 2) 
            final_score = round(total_qual * mult, 1) 

            with col_v:
                st.subheader("2. Quantitative Valuation")
                with st.container(border=True):
                    st.caption(f"Price ({data['currency']})"); st.metric("Price", f"{data['price']:.2f}", label_visibility="collapsed")
                    st.caption("PE Ratio"); st.metric("PE Ratio", f"{pe:.2f}", label_visibility="collapsed")
                    st.divider(); st.subheader("Valuation Multiplier")
                    st.markdown(f"""<div class="multiplier-box" style="border: 2px solid {color_code}; color: {color_code};">x{mult}</div>""", unsafe_allow_html=True)
                    if "Undervalued" in pe_text or "Fair" in pe_text: st.success(pe_text)
                    else: st.warning(pe_text)

            verdict_color = "#00C805" if final_score >= 75 else "#FFA500" if final_score >= 45 else "#FF0000"
            st.markdown(f"""<div class="final-score-box" style="border-color: {verdict_color};"><h2 style="color:#333;margin:0;">VALUE SCORE</h2><h1 style="color:{verdict_color};font-size:80px;margin:0;">{final_score}</h1></div>""", unsafe_allow_html=True)

        # ==========================================
        # TAB 2: TECHNICAL ANALYSIS
        # ==========================================
        with tab_tech:
            tech = calculate_technicals(data['history'])
            if tech:
                action = "WAIT / WATCH 🟡"; reason = "Market is indecisive."
                if "Uptrend" in tech['trend']:
                    if tech['last_price'] < tech['support'] * 1.05: action, reason = "BUY (Support Bounce) 🟢", "Uptrend + Near Support."
                    elif tech['vol_ratio'] > 1.5: action, reason = "STRONG BUY (Breakout) 🚀", "Uptrend + High Volume."
                    elif tech['is_squeezing']: action, reason = "PREPARE TO BUY (VCP) 🔵", "Volatility Squeeze detected."
                    elif tech['rsi'] > 70: action, reason = "HOLD / TAKE PROFIT 🟠", "Uptrend but Overbought."
                    else: action, reason = "BUY / HOLD 🟢", "Healthy Uptrend."
                else:
                    if tech['last_price'] < tech['support']: action, reason = "SELL / AVOID 🔴", "Breaking below Support."
                    elif tech['rsi'] < 30: action, reason = "WATCH (Oversold) 🟡", "Potential oversold bounce."
                    else: action, reason = "AVOID / SELL 🔴", "Stock is in a Downtrend."

                st.subheader(f"Technical Verdict: {action}"); st.info(f"📝 Reason: {reason}")
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Trend", tech['trend'])
                tc2.metric("RSI (14)", f"{tech['rsi']:.1f}", delta="High" if tech['rsi']>70 else "Low" if tech['rsi']<30 else "OK", delta_color="inverse")
                tc3.metric("Vol Ratio", f"{tech['vol_ratio']:.2f}x")
                tc4.metric("Squeeze", "YES" if tech['is_squeezing'] else "No")
                c_sup, c_res = st.columns(2)
                c_sup.success(f"🛡️ Support: {tech['support']:.2f}"); c_res.error(f"🚧 Resistance: {tech['resistance']:.2f}")
                st.line_chart(data['history'][['Close', 'SMA_50', 'SMA_200']], color=["#0000FF", "#FFA500", "#FF0000"]) 
            else: st.warning("Not enough historical data.")

        # ==========================================
        # TAB 3: FINANCIALS
        # ==========================================
        with tab_fin:
            i = data['raw_info']
            
            def make_row(cols):
                c = st.columns(len(cols))
                for idx, (label, val) in enumerate(cols):
                    c[idx].metric(label, val)

            st.caption("Key Fundamentals")
            
            # Row 1: Valuation
            make_row([
                ("Market Cap", fmt_num(i.get('marketCap'), is_currency=True)),
                ("Enterprise Val", fmt_num(i.get('enterpriseValue'), is_currency=True)),
                ("Trailing P/E", fmt_num(i.get('trailingPE'))),
                ("Forward P/E", fmt_num(i.get('forwardPE')))
            ])
            st.divider()
            
            # Row 2: Efficiency & Ratios
            make_row([
                ("PEG Ratio", fmt_num(i.get('pegRatio'))),
                ("Price/Sales", fmt_num(i.get('priceToSalesTrailing12Months'))),
                ("Price/Book", fmt_num(i.get('priceToBook'))),
                ("Beta", fmt_num(i.get('beta')))
            ])
            st.divider()
            
            # Row 3: Profitability (Usually decimals, so we multiply by 100 in fmt_num)
            make_row([
                ("Profit Margin", fmt_num(i.get('profitMargins'), is_pct=True)),
                ("Gross Margin", fmt_num(i.get('grossMargins'), is_pct=True)),
                ("ROA", fmt_num(i.get('returnOnAssets'), is_pct=True)),
                ("ROE", fmt_num(i.get('returnOnEquity'), is_pct=True))
            ])
            st.divider()
            
            # Row 4: Income & Dividend (Using specific fmt_dividend)
            make_row([
                ("EPS (ttm)", fmt_num(i.get('trailingEps'))),
                ("Revenue (ttm)", fmt_num(i.get('totalRevenue'), is_currency=True)),
                ("Dividend Yield", fmt_dividend(i.get('dividendYield'))),
                ("Target Price", fmt_num(i.get('targetMeanPrice')))
            ])
            
            st.markdown("---")
            st.caption(f"Fiscal Year End: {i.get('lastFiscalYearEnd', '-')}")

    else:
        st.error(f"Ticker '{final_t}' not found.")
