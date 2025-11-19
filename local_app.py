import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import re
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Local Value Investor", layout="wide", page_icon="📈")

# --- SESSION STATE & TRANSLATION SETUP ---
if 'language' not in st.session_state:
    st.session_state.language = 'EN'

def toggle_language():
    st.session_state.language = 'CN' if st.session_state.language == 'EN' else 'EN'

# --- TRANSLATION DICTIONARY (TRADITIONAL CHINESE STRICT) ---
T = {
    "EN": {
        "app_title": "Local Value Investor",
        "sidebar_title": "Analysis Tool",
        "market_label": "Select Market",
        "ticker_label": "Enter Stock Ticker",
        "analyze_btn": "Analyze Stock",
        "analyze_mobile_btn": "Analyze (Mobile)",
        "connected": "🟢 LM Studio Connected",
        "disconnected": "🔴 LM Studio Disconnected",
        "methodology": "Methodology:",
        "qual_score": "Qualitative Score (0-20)",
        "qual_detail": "(5 topics x 4 pts)",
        "val_mult": "Valuation Multiplier (1-5)",
        "val_detail": "(Based on PE Ratio)",
        "final_score": "= Final Score (0-100)",
        "tab_value": "💎 Value Analysis",
        "tab_tech": "📈 Technical Analysis",
        "tab_fin": "📊 Financials",
        "topics": [
            "Unique Product/Moat", "Revenue Growth", "Competitive Advantage", "Profit Stability", "Management"
        ],
        "loading_data": "Fetching data for",
        "loading_ai": "AI Analyzing:",
        "currency": "Currency",
        "industry": "Industry",
        "val_analysis_header": "1. Qualitative Analysis",
        "quant_val_header": "2. Quantitative Valuation",
        "price": "Price",
        "pe_ratio": "PE Ratio",
        "multiplier_label": "Valuation Multiplier",
        "verdict_buy": "BUY",
        "verdict_sell": "SELL",
        "verdict_hold": "HOLD",
        "tech_verdict": "Technical Verdict",
        "reason": "Reason",
        "support": "Support",
        "resistance": "Resistance",
        "trend": "Trend",
        "squeeze": "Squeeze",
        "recent_div": "💰 Recent Dividend History (Last 10)",
        "no_div": "No recent dividend history available.",
        "fiscal_year": "Fiscal Year End",
        # Financial Table Labels
        "fin_mkt_cap": "Market Cap", "fin_ent_val": "Enterprise Val",
        "fin_trail_pe": "Trailing P/E", "fin_fwd_pe": "Forward P/E",
        "fin_peg": "PEG Ratio", "fin_ps": "Price/Sales",
        "fin_pb": "Price/Book", "fin_beta": "Beta",
        "fin_prof_marg": "Profit Margin", "fin_gross_marg": "Gross Margin",
        "fin_roa": "ROA", "fin_roe": "ROE",
        "fin_eps": "EPS (ttm)", "fin_rev": "Revenue (ttm)",
        "fin_div_yield": "Dividend Yield", "fin_target": "Target Price",
        # PE Text
        "pe_neg": "❌ Negative / No Earnings",
        "pe_under": "✅ Undervalued (PE < 20)",
        "pe_over": "⚠️ Overvalued (PE > 75)",
        "pe_fair": "✅ Fairly Valued",
        "pe_ok": "⚖️ Fair Value",
        "pe_exp": "⚠️ Expensive",
        # Technical Logic Text
        "uptrend": "Uptrend", "downtrend": "Downtrend",
        "weak_uptrend": "Weak Uptrend", "neutral": "Neutral",
        "act_buy_sup": "BUY (Support Bounce) 🟢",
        "act_buy_break": "STRONG BUY (Breakout) 🚀",
        "act_prep": "PREPARE TO BUY (VCP) 🔵",
        "act_profit": "HOLD / TAKE PROFIT 🟠",
        "act_buy_hold": "BUY / HOLD 🟢",
        "act_sell_sup": "SELL / AVOID 🔴",
        "act_watch_oversold": "WATCH (Oversold) 🟡",
        "act_avoid": "AVOID / SELL 🔴",
        # Reasons
        "reas_sup": "Uptrend + Near Support.",
        "reas_vol": "Uptrend + High Volume.",
        "reas_vcp": "Volatility Squeeze detected.",
        "reas_over": "Uptrend but Overbought.",
        "reas_health": "Healthy Uptrend.",
        "reas_break_sup": "Breaking below Support.",
        "reas_oversold": "Potential oversold bounce.",
        "reas_down": "Stock is in a Downtrend."
    },
    "CN": {
        "app_title": "本地價值投資助手",
        "sidebar_title": "股票分析工具",
        "market_label": "選擇市場",
        "ticker_label": "輸入股票代號",
        "analyze_btn": "開始分析",
        "analyze_mobile_btn": "開始分析 (手機版)",
        "connected": "🟢 LM Studio 已連接",
        "disconnected": "🔴 LM Studio 未連接",
        "methodology": "分析方法:",
        "qual_score": "定性評分 (0-20)",
        "qual_detail": "(5個主題 x 4分)",
        "val_mult": "估值倍數 (1-5)",
        "val_detail": "(基於市盈率 PE)",
        "final_score": "= 最終評分 (0-100)",
        "tab_value": "💎 價值分析",
        "tab_tech": "📈 技術分析",
        "tab_fin": "📊 財務數據",
        "topics": [
            "獨特產品/護城河", "營收增長潛力", "競爭優勢", "獲利穩定性", "管理層質素"
        ],
        "loading_data": "正在獲取數據：",
        "loading_ai": "AI 正在分析：",
        "currency": "貨幣",
        "industry": "行業",
        "val_analysis_header": "1. 定性分析 (AI)",
        "quant_val_header": "2. 量化估值",
        "price": "股價",
        "pe_ratio": "市盈率 (PE)",
        "multiplier_label": "估值倍數",
        "verdict_buy": "買入",
        "verdict_sell": "賣出",
        "verdict_hold": "持有",
        "tech_verdict": "技術面結論",
        "reason": "理由",
        "support": "支持位",
        "resistance": "阻力位",
        "trend": "趨勢",
        "squeeze": "擠壓 (VCP)",
        "recent_div": "💰 近期派息記錄 (最近10次)",
        "no_div": "沒有近期派息記錄。",
        "fiscal_year": "財政年度結算日",
        # Financial Table Labels
        "fin_mkt_cap": "市值", "fin_ent_val": "企業價值",
        "fin_trail_pe": "歷史市盈率", "fin_fwd_pe": "預測市盈率",
        "fin_peg": "PEG 比率", "fin_ps": "市銷率 (P/S)",
        "fin_pb": "市賬率 (P/B)", "fin_beta": "Beta 系數",
        "fin_prof_marg": "淨利潤率", "fin_gross_marg": "毛利率",
        "fin_roa": "資產回報率 (ROA)", "fin_roe": "股本回報率 (ROE)",
        "fin_eps": "每股盈利 (EPS)", "fin_rev": "總營收",
        "fin_div_yield": "股息率", "fin_target": "目標價",
        # PE Text
        "pe_neg": "❌ 負收益 / 無盈利",
        "pe_under": "✅ 被低估 (PE < 20)",
        "pe_over": "⚠️ 被高估 (PE > 75)",
        "pe_fair": "✅ 估值合理",
        "pe_ok": "⚖️ 估值適中",
        "pe_exp": "⚠️ 估值偏高",
        # Technical Logic Text
        "uptrend": "上升趨勢", "downtrend": "下降趨勢",
        "weak_uptrend": "弱勢上升", "neutral": "中性",
        "act_buy_sup": "買入 (支持位反彈) 🟢",
        "act_buy_break": "強力買入 (突破) 🚀",
        "act_prep": "準備買入 (VCP擠壓) 🔵",
        "act_profit": "持有 / 獲利止盈 🟠",
        "act_buy_hold": "買入 / 持有 🟢",
        "act_sell_sup": "賣出 / 觀望 🔴",
        "act_watch_oversold": "關注 (超賣反彈) 🟡",
        "act_avoid": "觀望 / 賣出 🔴",
        # Reasons
        "reas_sup": "上升趨勢 + 接近支持位。",
        "reas_vol": "上升趨勢 + 成交量激增。",
        "reas_vcp": "檢測到波動率擠壓 (VCP)。",
        "reas_over": "上升趨勢但超買。",
        "reas_health": "健康的上升趨勢。",
        "reas_break_sup": "跌破支持位。",
        "reas_oversold": "下跌趨勢但可能超賣反彈。",
        "reas_down": "股價處於下降趨勢。"
    }
}

# Helper to get text based on current language
def txt(key):
    return T[st.session_state.language][key]

# --- CSS STYLING ---
st.markdown("""
<style>
    .multiplier-box {
        font-size: 35px; font-weight: bold; text-align: center; padding: 15px; 
        border-radius: 10px; background-color: #ffffff; margin-top: 10px;
        margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .methodology-box {
        background-color: #262730; padding: 15px; border-radius: 10px;
        border: 1px solid #444; font-size: 14px; margin-top: 20px;
    }
    .final-score-box {
        text-align: center; padding: 20px; border-radius: 15px; 
        background-color: #ffffff; margin-top: 20px; border: 4px solid #ccc;
    }
    div[data-testid="stMetricValue"] { font-size: 18px !important; }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; color: #888; }
    div[data-testid="stForm"] button[kind="primary"] {
        background-color: #FF4B4B; color: white; border: none;
        font-weight: bold; font-size: 16px; padding: 0.5rem 1rem; width: 100%;
    }
    div[data-testid="stForm"] button[kind="primary"]:hover {
        background-color: #FF0000; border-color: #FF0000;
    }
    /* Language Button Style */
    .lang-btn { margin-top: 0px; }
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
    if val is None or val == "N/A": return "-"
    if is_pct: return f"{val * 100:.2f}%"
    if is_currency:
        if val > 1e12: return f"{val/1e12:.2f}T"
        if val > 1e9: return f"{val/1e9:.2f}B"
        if val > 1e6: return f"{val/1e6:.2f}M"
    return f"{val:.2f}"

def fmt_dividend(val):
    if val is None: return "-"
    return f"{val:.2f}%"

def fmt_date(ts):
    """Converts Unix timestamp to YYYY-MM-DD"""
    if ts is None: return "-"
    try:
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    except:
        return str(ts)

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info: return None
        price = info.get('currentPrice', 0)
        hist = stock.history(period="1y")
        if price == 0 and not hist.empty: price = hist['Close'].iloc[-1]
        
        pe = info.get('trailingPE')
        if pe is None or pe == 'N/A':
            eps = info.get('forwardEps', info.get('trailingEps', 0))
            pe = price / eps if eps and eps > 0 else 0
        
        divs = stock.dividends

        return {
            "price": price, "currency": info.get('currency', 'USD'), "pe": pe,
            "name": info.get('longName', ticker), "industry": info.get('industry', 'Unknown'),
            "summary": info.get('longBusinessSummary', 'No summary available.'), 
            "history": hist, "dividends": divs, "raw_info": info 
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
    
    trend = "neutral"
    if current_price > sma_200:
        trend = "uptrend" if current_price > sma_50 else "weak_uptrend"
    else:
        trend = "downtrend"
    return {
        "trend": trend, "rsi": rsi, "support": support, "resistance": resistance,
        "vol_ratio": vol_ratio, "is_squeezing": is_squeezing,
        "sma_50": sma_50, "sma_200": sma_200, "last_price": current_price
    }

def analyze_qualitative(ticker, summary, topic):
    # --- STRONGER PROMPT FOR CHINESE ---
    if st.session_state.language == 'CN':
        system_role = "You are a strict financial analyst. You MUST output in Traditional Chinese (繁體中文)."
        lang_instruction = (
            "IMPORTANT: The Context provided is in English, but your analysis and reason MUST be written in Traditional Chinese (繁體中文). "
            "Do NOT write the reason in English. Translate your thoughts."
            "\n\nExample Output format: 3.5|該公司擁有強大的品牌優勢，且現金流穩定。"
        )
    else:
        system_role = "You are a strict financial analyst."
        lang_instruction = "Answer in English."
    
    prompt = (
        f"Analyze {ticker} regarding '{topic}'. "
        f"Context: {summary}. "
        f"Give a specific score from 0.0 to 4.0 (use 1 decimal place). "
        f"Provide a 1 sentence reason. {lang_instruction} "
        f"Strict Format: SCORE|REASON"
    )
    
    try:
        resp = client.chat.completions.create(
            model="local-model", 
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, max_tokens=800
        )
        raw_content = resp.choices[0].message.content
        clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        return clean_content, False
    except Exception as e:
        return f"0.0|Error: {str(e)}", True

# --- TOP LAYOUT & LANGUAGE TOGGLE ---
top_col1, top_col2 = st.columns([8, 1])
with top_col2:
    if st.button("🌐 Eng / 中"):
        toggle_language()
        st.rerun()

# --- INPUT LOGIC ---
if 'layout_mode' not in st.session_state: st.session_state.layout_mode = 'desktop' 
if 'active_ticker' not in st.session_state: st.session_state.active_ticker = "NVDA"
if 'active_market' not in st.session_state: st.session_state.active_market = "US"

# --- SIDEBAR ---
with st.sidebar:
    st.header(txt('sidebar_title'))
    with st.form(key='desktop_form'):
        st.caption(txt('market_label'))
        d_market = st.selectbox("Market", ["US", "Canada (TSX)", "HK (HKEX)"], label_visibility="collapsed")
        st.caption(txt('ticker_label'))
        d_ticker = st.text_input("Ticker", value="NVDA", label_visibility="collapsed").upper()
        d_submit = st.form_submit_button(txt('analyze_btn'), type="primary") 
    
    st.markdown("---")
    
    if connection_status:
        try:
            client.models.list()
            st.success(txt('connected'))
        except: st.error(txt('disconnected'))
    else: st.error(txt('disconnected'))

    st.markdown(f"""
<div class="methodology-box">
<h4 style="margin-top:0; color: #4da6ff;">{txt('methodology')}</h4>
<p style="margin-bottom: 5px;"><strong style="color: #4da6ff;">{txt('qual_score')}</strong><br>
<span style="color: #aaa; font-size: 12px;">{txt('qual_detail')}</span></p>
<p style="text-align:center; margin: 5px 0;">✖</p>
<p style="margin-bottom: 5px;"><strong style="color: #4da6ff;">{txt('val_mult')}</strong><br>
<span style="color: #aaa; font-size: 12px;">{txt('val_detail')}</span></p>
<hr style="margin: 10px 0; border-color: #555;">
<p style="margin-bottom: 0;"><strong style="color: #4da6ff;">{txt('final_score')}</strong></p>
</div>
""", unsafe_allow_html=True)

# --- MOBILE SEARCH ---
with st.expander(f"📱 {txt('analyze_mobile_btn')}", expanded=False):
    with st.form(key='mobile_form'):
        m_col1, m_col2 = st.columns([1, 1])
        with m_col1: m_market = st.selectbox(txt('market_label'), ["US", "Canada (TSX)", "HK (HKEX)"], key='m_m')
        with m_col2: m_ticker = st.text_input(txt('ticker_label'), value="NVDA", key='m_t').upper()
        m_submit = st.form_submit_button(txt('analyze_mobile_btn'), type="primary")

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

    with st.spinner(f"{txt('loading_data')} {final_t}..."):
        data = get_stock_data(final_t)

    if data:
        st.header(f"{data['name']} ({final_t})")
        st.caption(f"{txt('industry')}: {data['industry']} | {txt('currency')}: {data['currency']}")
        
        tab_fund, tab_tech, tab_fin = st.tabs([txt('tab_value'), txt('tab_tech'), txt('tab_fin')])

        # ==========================================
        # TAB 1: FUNDAMENTAL VALUE
        # ==========================================
        with tab_fund:
            english_topics = ["Unique Product/Moat", "Revenue Growth", "Competitive Advantage", "Profit Stability", "Management"]
            translated_topics = txt('topics')
            
            qual_results = []
            total_qual = 0.0 
            prog_bar = st.progress(0)
            status_text = st.empty()
            
            col_q, col_v = st.columns([1.6, 1])
            
            with col_q:
                st.subheader(txt('val_analysis_header'))
                for i, t_eng in enumerate(english_topics):
                    t_display = translated_topics[i]
                    prog_bar.progress((i)/5)
                    status_text.text(f"{txt('loading_ai')} {t_display}...")
                    
                    res, is_error = analyze_qualitative(data['name'], data['summary'], t_eng)
                    
                    match = re.search(r'\b([0-3](?:\.\d)?|4(?:\.0)?)\b', res)
                    if match:
                        s_str = match.group(1); s = float(s_str)
                        r = res.replace(s_str, "").replace("|", "").replace("SCORE", "").replace("REASON", "").strip().strip(' :-=\n')
                    else: s, r = 0.0, res 
                    total_qual += s
                    qual_results.append((t_display, s, r))
                    with st.expander(f"{t_display}", expanded=True): st.markdown(f"**{s}/4** — {r}")

                prog_bar.empty(); status_text.empty()

            pe = data['pe']
            if pe is None or pe <= 0: mult, color_code, pe_text = 1.0, "#FF4500", txt('pe_neg')
            elif pe <= 20: mult, color_code, pe_text = 5.0, "#00C805", txt('pe_under')
            elif pe >= 75: mult, color_code, pe_text = 1.0, "#FF4500", txt('pe_over')
            else:
                pct = (pe - 20) / 55; mult = 5.0 - (pct * 4.0)
                if mult >= 4.0: color_code, pe_text = "#00C805", txt('pe_fair')
                elif mult >= 3.0: color_code, pe_text = "#90EE90", txt('pe_ok')
                elif mult >= 2.0: color_code, pe_text = "#FFA500", txt('pe_exp')
                else: color_code, pe_text = "#FF4500", txt('pe_exp')

            mult = round(mult, 2) 
            final_score = round(total_qual * mult, 1) 

            with col_v:
                st.subheader(txt('quant_val_header'))
                with st.container(border=True):
                    st.caption(f"{txt('price')} ({data['currency']})"); st.metric("Price", f"{data['price']:.2f}", label_visibility="collapsed")
                    st.caption(txt('pe_ratio')); st.metric("PE Ratio", f"{pe:.2f}" if pe else "N/A", label_visibility="collapsed")
                    st.divider(); st.subheader(txt('multiplier_label'))
                    st.markdown(f"""<div class="multiplier-box" style="border: 2px solid {color_code}; color: {color_code};">x{mult}</div>""", unsafe_allow_html=True)
                    if color_code in ["#00C805", "#90EE90"]: st.success(pe_text)
                    else: st.warning(pe_text)

            verdict_color = "#00C805" if final_score >= 75 else "#FFA500" if final_score >= 45 else "#FF0000"
            st.markdown(f"""<div class="final-score-box" style="border-color: {verdict_color};"><h2 style="color:#333;margin:0;">VALUE SCORE</h2><h1 style="color:{verdict_color};font-size:80px;margin:0;">{final_score}</h1></div>""", unsafe_allow_html=True)

        # ==========================================
        # TAB 2: TECHNICAL ANALYSIS
        # ==========================================
        with tab_tech:
            tech = calculate_technicals(data['history'])
            if tech:
                action_key = "act_avoid" 
                reason_key = "neutral"
                
                if "uptrend" in tech['trend']:
                    if tech['last_price'] < tech['support'] * 1.05: action_key, reason_key = "act_buy_sup", "reas_sup"
                    elif tech['vol_ratio'] > 1.5: action_key, reason_key = "act_buy_break", "reas_vol"
                    elif tech['is_squeezing']: action_key, reason_key = "act_prep", "reas_vcp"
                    elif tech['rsi'] > 70: action_key, reason_key = "act_profit", "reas_over"
                    else: action_key, reason_key = "act_buy_hold", "reas_health"
                else:
                    if tech['last_price'] < tech['support']: action_key, reason_key = "act_sell_sup", "reas_break_sup"
                    elif tech['rsi'] < 30: action_key, reason_key = "act_watch_oversold", "reas_oversold"
                    else: action_key, reason_key = "act_avoid", "reas_down"

                st.subheader(f"{txt('tech_verdict')}: {txt(action_key)}"); st.info(f"📝 {txt('reason')}: {txt(reason_key)}")
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric(txt('trend'), txt(tech['trend']))
                tc2.metric("RSI (14)", f"{tech['rsi']:.1f}", delta="High" if tech['rsi']>70 else "Low" if tech['rsi']<30 else "OK", delta_color="inverse")
                tc3.metric("Vol Ratio", f"{tech['vol_ratio']:.2f}x")
                tc4.metric(txt('squeeze'), "YES" if tech['is_squeezing'] else "No")
                c_sup, c_res = st.columns(2)
                c_sup.success(f"🛡️ {txt('support')}: {tech['support']:.2f}"); c_res.error(f"🚧 {txt('resistance')}: {tech['resistance']:.2f}")
                st.line_chart(data['history'][['Close', 'SMA_50', 'SMA_200']], color=["#0000FF", "#FFA500", "#FF0000"]) 
            else: st.warning("Not enough historical data.")

        # ==========================================
        # TAB 3: FINANCIALS
        # ==========================================
        with tab_fin:
            i = data['raw_info']
            def make_row(cols):
                c = st.columns(len(cols))
                for idx, (label_key, val) in enumerate(cols):
                    c[idx].metric(txt(label_key), val)

            st.caption(txt('tab_fin'))
            make_row([("fin_mkt_cap", fmt_num(i.get('marketCap'), is_currency=True)), ("fin_ent_val", fmt_num(i.get('enterpriseValue'), is_currency=True)), ("fin_trail_pe", fmt_num(i.get('trailingPE'))), ("fin_fwd_pe", fmt_num(i.get('forwardPE')))])
            st.divider()
            make_row([("fin_peg", fmt_num(i.get('pegRatio'))), ("fin_ps", fmt_num(i.get('priceToSalesTrailing12Months'))), ("fin_pb", fmt_num(i.get('priceToBook'))), ("fin_beta", fmt_num(i.get('beta')))])
            st.divider()
            make_row([("fin_prof_marg", fmt_num(i.get('profitMargins'), is_pct=True)), ("fin_gross_marg", fmt_num(i.get('grossMargins'), is_pct=True)), ("fin_roa", fmt_num(i.get('returnOnAssets'), is_pct=True)), ("fin_roe", fmt_num(i.get('returnOnEquity'), is_pct=True))])
            st.divider()
            make_row([("fin_eps", fmt_num(i.get('trailingEps'))), ("fin_rev", fmt_num(i.get('totalRevenue'), is_currency=True)), ("fin_div_yield", fmt_dividend(i.get('dividendYield'))), ("fin_target", fmt_num(i.get('targetMeanPrice')))])
            
            st.markdown("---")
            st.subheader(txt('recent_div'))
            divs = data.get('dividends')
            if divs is not None and not divs.empty:
                divs_sorted = divs.sort_index(ascending=False).head(10)
                df_divs = divs_sorted.reset_index()
                df_divs.columns = ["Date", "Amount"]
                df_divs['Date'] = df_divs['Date'].dt.strftime('%Y-%m-%d')
                df_divs['Amount'] = df_divs['Amount'].apply(lambda x: f"{data['currency']} {x:.4f}")
                st.table(df_divs)
            else:
                st.info(txt('no_div'))
            
            st.caption(f"{txt('fiscal_year')}: {fmt_date(i.get('lastFiscalYearEnd'))}")

    else:
        st.error(f"Ticker '{final_t}' not found.")
