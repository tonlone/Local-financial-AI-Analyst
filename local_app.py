import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import re
from datetime import datetime
import pytz

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Local Value Investor", layout="wide", page_icon="📈")

# --- SESSION STATE & TRANSLATION SETUP ---
if 'language' not in st.session_state:
    st.session_state.language = 'EN'

def toggle_language():
    st.session_state.language = 'CN' if st.session_state.language == 'EN' else 'EN'

# --- TRANSLATION DICTIONARY ---
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
        "val_detail": "(Based on Forward PE Range)",
        "final_score": "= Final Score (0-100)",
        "tab_value": "💎 Value Analysis",
        "tab_tech": "📈 Technical Analysis",
        "tab_fin": "📊 Financials",
        "tab_news": "📰 News & Earnings",
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
        "pe_ttm": "Trailing PE (TTM)",
        "pe_ratio": "Forward PE (Used for Calc)",
        "multiplier_label": "Valuation Multiplier",
        
        # Score Calculation
        "calc_qual": "Qualitative Score",
        "calc_mult": "Multiplier",
        "calc_result": "Final Score",
        "score_calc_title": "VALUE SCORE CALCULATION",

        # Valuation Specifics
        "hist_low_pe": "Hist. Low PE (5Y)",
        "hist_high_pe": "Hist. High PE (5Y)",
        "pe_pos": "PE Position (5Y)",
        "pe_pos_low": "Low (Cheap)",
        "pe_pos_high": "High (Expensive)",
        "grade_strong_buy": "Very Excellent / Strong Buy",
        "grade_buy": "Excellent / Buy",
        "grade_hold": "Good / Hold",
        "grade_sell": "Average / Sell",
        "grade_avoid": "Poor / Avoid",
        "grading_scale": "Grading Scale:",
        
        "verdict_buy": "BUY", "verdict_sell": "SELL", "verdict_hold": "HOLD",
        "tech_verdict": "Technical Verdict", "reason": "Reason",
        "support": "Support", "resistance": "Resistance", "trend": "Trend", "squeeze": "Squeeze",
        "recent_div": "💰 Recent Dividend History (Last 10)",
        "no_div": "No recent dividend history available.",
        "fiscal_year": "Fiscal Year End",
        "fin_mkt_cap": "Market Cap", "fin_ent_val": "Enterprise Val",
        "fin_trail_pe": "Trailing P/E", "fin_fwd_pe": "Forward P/E",
        "fin_peg": "PEG Ratio", "fin_ps": "Price/Sales",
        "fin_pb": "Price/Book", "fin_beta": "Beta",
        "fin_prof_marg": "Profit Margin", "fin_gross_marg": "Gross Margin",
        "fin_roa": "ROA", "fin_roe": "ROE",
        "fin_eps": "EPS (ttm)", "fin_rev": "Revenue (ttm)",
        "fin_div_yield": "Dividend Yield", "fin_target": "Target Price",
        "pe_neg": "❌ Negative / No Earnings",
        "uptrend": "Uptrend", "downtrend": "Downtrend", "weak_uptrend": "Weak Uptrend", "neutral": "Neutral",
        "act_buy_sup": "BUY (Support Bounce) 🟢", "act_buy_break": "STRONG BUY (Breakout) 🚀",
        "act_prep": "PREPARE TO BUY (VCP) 🔵", "act_profit": "HOLD / TAKE PROFIT 🟠",
        "act_buy_hold": "BUY / HOLD 🟢", "act_sell_sup": "SELL / AVOID 🔴",
        "act_watch_oversold": "WATCH (Oversold) 🟡", "act_avoid": "AVOID / SELL 🔴",
        "reas_sup": "Uptrend + Near Support.", "reas_vol": "Uptrend + High Volume.",
        "reas_vcp": "Volatility Squeeze detected.", "reas_over": "Uptrend but Overbought.",
        "reas_health": "Healthy Uptrend.", "reas_break_sup": "Breaking below Support.",
        "reas_oversold": "Potential oversold bounce.", "reas_down": "Stock is in a Downtrend.",
        "lbl_rsi": "RSI (14)", "lbl_vol": "Vol Ratio",
        "status_high": "High", "status_low": "Low", "status_ok": "OK",
        
        # Earnings Tab
        "earn_title": "Latest Earnings Announcement",
        "earn_date": "Date",
        "earn_est_eps": "Est. EPS",
        "earn_act_eps": "Actual EPS",
        "earn_surprise": "Surprise",
        "ai_summary_title": "AI Earnings & News Summary",
        "source_link": "Search Official Earnings Report",
        
        # NEW Q/Q Financials
        "qq_title": "Quarterly Financial Trends (Q/Q Change)",
        "qq_rev": "Revenue",
        "qq_net_inc": "Net Income",
        "qq_eps": "Net Income / Share",
        "qq_op_inc": "Operating Income",
        "qq_op_exp": "Operating Expenses",
        "qq_gross_marg": "Gross Margin",
        
        # Multiplier Explanation
        "mult_how": "❓ How is this calculated?",
        "mult_exp_title": "Logic: Buy Low, Sell High",
        "mult_exp_desc": "We compare the current PE to its 5-year range. Lower PE (Cheap) gets a higher multiplier to boost the score.",
        "mult_formula": "Position Formula:",
        "mult_table_pos": "PE Position",
        "mult_table_mult": "Multiplier",
        "mult_table_mean": "Meaning",
        "status_under": "Undervalued",
        "status_fair": "Fair Value",
        "status_over": "Overvalued",
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
        "val_detail": "(基於預測 PE 區間)",
        "final_score": "= 最終評分 (0-100)",
        "tab_value": "💎 價值分析",
        "tab_tech": "📈 技術分析",
        "tab_fin": "📊 財務數據",
        "tab_news": "📰 新聞與財報",
        "topics": ["獨特產品/護城河", "營收增長潛力", "競爭優勢", "獲利穩定性", "管理層質素"],
        "loading_data": "正在獲取數據：",
        "loading_ai": "AI 正在分析：",
        "currency": "貨幣",
        "industry": "行業",
        "val_analysis_header": "1. 定性分析 (AI)",
        "quant_val_header": "2. 量化估值",
        "price": "當前股價",
        "pe_ttm": "歷史市盈率 (Trailing PE)",
        "pe_ratio": "預測市盈率 (Forward PE)",
        "multiplier_label": "本益比乘數 (Multiplier)",
        
        # Score Calculation
        "calc_qual": "投資評估分數",
        "calc_mult": "本益比乘數",
        "calc_result": "最終評分",
        "score_calc_title": "價值評分計算",

        # Valuation Specifics
        "hist_low_pe": "歷史最低 PE (5年)",
        "hist_high_pe": "歷史最高 PE (5年)",
        "pe_pos": "目前 PE 位置區間",
        "pe_pos_low": "低位 (便宜)",
        "pe_pos_high": "高位 (昂貴)",
        "grade_strong_buy": "非常優秀 (Strong Buy)",
        "grade_buy": "優秀 (Buy)",
        "grade_hold": "良好 (Hold)",
        "grade_sell": "普通 (Sell)",
        "grade_avoid": "差 (Avoid)",
        "grading_scale": "評級標準:",
        
        "verdict_buy": "買入", "verdict_sell": "賣出", "verdict_hold": "持有",
        "tech_verdict": "技術面結論", "reason": "理由",
        "support": "支持位", "resistance": "阻力位", "trend": "趨勢", "squeeze": "擠壓 (VCP)",
        "recent_div": "💰 近期派息記錄 (最近10次)",
        "no_div": "沒有近期派息記錄。",
        "fiscal_year": "財政年度結算日",
        "fin_mkt_cap": "市值", "fin_ent_val": "企業價值",
        "fin_trail_pe": "歷史市盈率", "fin_fwd_pe": "預測市盈率",
        "fin_peg": "PEG 比率", "fin_ps": "市銷率 (P/S)",
        "fin_pb": "市賬率 (P/B)", "fin_beta": "Beta 系數",
        "fin_prof_marg": "淨利潤率", "fin_gross_marg": "毛利率",
        "fin_roa": "資產回報率 (ROA)", "fin_roe": "股本回報率 (ROE)",
        "fin_eps": "每股盈利 (EPS)", "fin_rev": "總營收",
        "fin_div_yield": "股息率", "fin_target": "目標價",
        "pe_neg": "❌ 負收益 / 無盈利",
        "uptrend": "上升趨勢", "downtrend": "下降趨勢", "weak_uptrend": "弱勢上升", "neutral": "中性",
        "act_buy_sup": "買入 (支持位反彈) 🟢", "act_buy_break": "強力買入 (突破) 🚀",
        "act_prep": "準備買入 (VCP擠壓) 🔵", "act_profit": "持有 / 獲利止盈 🟠",
        "act_buy_hold": "買入 / 持有 🟢", "act_sell_sup": "賣出 / 觀望 🔴",
        "act_watch_oversold": "關注 (超賣反彈) 🟡", "act_avoid": "觀望 / 賣出 🔴",
        "reas_sup": "上升趨勢 + 接近支持位。", "reas_vol": "上升趨勢 + 成交量激增。",
        "reas_vcp": "檢測到波動率擠壓 (VCP)。", "reas_over": "上升趨勢但超買。",
        "reas_health": "健康的上升趨勢。", "reas_break_sup": "跌破支持位。",
        "reas_oversold": "下跌趨勢但可能超賣反彈。", "reas_down": "股價處於下降趨勢。",
        "lbl_rsi": "相對強弱指數", "lbl_vol": "成交量比率",
        "status_high": "偏高", "status_low": "偏低", "status_ok": "適中",
        
        # Earnings Tab
        "earn_title": "最新財報發布 (Earnings)",
        "earn_date": "發布日期",
        "earn_est_eps": "預估 EPS",
        "earn_act_eps": "實際 EPS",
        "earn_surprise": "驚喜幅度 (Surprise)",
        "ai_summary_title": "AI 財報與新聞摘要",
        "source_link": "搜尋官方財報",
        
        # NEW Q/Q Financials
        "qq_title": "季度財務趨勢 (Q/Q 環比)",
        "qq_rev": "總營收 (Revenue)",
        "qq_net_inc": "淨利潤 (Net Income)",
        "qq_eps": "每股淨收益 (Net Income/Share)",
        "qq_op_inc": "營業利潤 (Op Income)",
        "qq_op_exp": "營業費用 (Op Expenses)",
        "qq_gross_marg": "毛利率 (Gross Margin)",
        
        # Multiplier Explanation
        "mult_how": "❓ 如何計算此倍數？",
        "mult_exp_title": "邏輯：低買高賣",
        "mult_exp_desc": "我們將當前 PE 與過去 5 年的歷史區間進行比較。PE 越低（便宜）則倍數越高，從而提升評分。",
        "mult_formula": "位置計算公式：",
        "mult_table_pos": "PE 區間位置",
        "mult_table_mult": "倍數 (Multiplier)",
        "mult_table_mean": "含義",
        "status_under": "被低估 (便宜)",
        "status_fair": "合理估值",
        "status_over": "被高估 (昂貴)",
    }
}

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
    /* Grade Table */
    .grade-table { width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 10px; }
    .grade-table td { padding: 5px; border: 1px solid #eee; text-align: center; }
    .grade-green { background-color: #e6ffe6; color: #006600; font-weight: bold; }
    .grade-lightgreen { background-color: #f0fff0; color: #009900; }
    .grade-yellow { background-color: #fffff0; color: #b3b300; }
    .grade-orange { background-color: #fff5e6; color: #cc6600; }
    .grade-red { background-color: #ffe6e6; color: #cc0000; }
    
    div[data-testid="stMetricValue"] { font-size: 18px !important; }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; color: #888; }
    div[data-testid="stForm"] button[kind="primary"] {
        background-color: #FF4B4B; color: white; border: none; font-weight: bold; font-size: 16px; padding: 0.5rem 1rem; width: 100%;
    }
    div[data-testid="stForm"] button[kind="primary"]:hover { background-color: #FF0000; border-color: #FF0000; }
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
    if ts is None: return "-"
    try: return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    except: return str(ts)

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info: return None
        
        price = info.get('currentPrice', 0)
        hist = stock.history(period="5y")
        if price == 0 and not hist.empty: price = hist['Close'].iloc[-1]
        
        eps = info.get('forwardEps')
        pe = info.get('forwardPE')
        if eps is None: eps = info.get('trailingEps')
        if pe is None: pe = info.get('trailingPE')
        if pe is None: pe = price / eps if (eps and eps > 0) else 0

        min_pe = 0; max_pe = 0
        if eps and eps > 0 and not hist.empty:
            pe_series = hist['Close'] / eps
            min_pe = pe_series.min(); max_pe = pe_series.max()
        
        divs = stock.dividends
        
        try: earnings_dates = stock.earnings_dates
        except: earnings_dates = None
        try: quarterly_financials = stock.quarterly_income_stmt
        except: quarterly_financials = None
            
        try: 
            raw_news = stock.news
            news = [n for n in raw_news if n.get('title')]
        except: 
            news = []

        return {
            "price": price, "currency": info.get('currency', 'USD'), "pe": pe,
            "eps": eps, "min_pe": min_pe, "max_pe": max_pe,
            "name": info.get('longName', ticker), "industry": info.get('industry', 'Unknown'),
            "summary": info.get('longBusinessSummary', 'No summary available.'), 
            "history": hist, "dividends": divs, "raw_info": info,
            "earnings_dates": earnings_dates, "quarterly_financials": quarterly_financials, "news": news
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
    if current_price > sma_200: trend = "uptrend" if current_price > sma_50 else "weak_uptrend"
    else: trend = "downtrend"
    return {
        "trend": trend, "rsi": rsi, "support": support, "resistance": resistance,
        "vol_ratio": vol_ratio, "is_squeezing": is_squeezing,
        "sma_50": sma_50, "sma_200": sma_200, "last_price": current_price
    }

def analyze_qualitative(ticker, summary, topic):
    if st.session_state.language == 'CN':
        system_role = "You are a strict financial analyst. You MUST output in Traditional Chinese (繁體中文)."
        lang_instruction = "IMPORTANT: The Context provided is in English, but your analysis and reason MUST be written in Traditional Chinese (繁體中文). Do NOT write the reason in English."
    else:
        system_role = "You are a strict financial analyst."
        lang_instruction = "Answer in English."
    
    if topic == "EarningsSummary":
        prompt = f"Summarize the recent financial performance and news for {ticker}. Context: {summary}. Keep it concise (3-4 bullet points). {lang_instruction}"
    else:
        prompt = f"Analyze {ticker} regarding '{topic}'. Context: {summary}. Give a specific score from 0.0 to 4.0 (use 1 decimal place). Provide a 1 sentence reason. {lang_instruction} Strict Format: SCORE|REASON"
    
    try:
        resp = client.chat.completions.create(
            model="local-model", 
            messages=[{"role": "system", "content": system_role}, {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=800
        )
        raw = resp.choices[0].message.content
        return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip(), False
    except Exception as e:
        return f"Error: {str(e)}", True

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
        try: client.models.list(); st.success(txt('connected'))
        except: st.error(txt('disconnected'))
    else: st.error(txt('disconnected'))

    st.markdown(f"""<div class="methodology-box"><h4 style="margin-top:0; color: #4da6ff;">{txt('methodology')}</h4><p style="margin-bottom: 5px;"><strong style="color: #4da6ff;">{txt('qual_score')}</strong><br><span style="color: #aaa; font-size: 12px;">{txt('qual_detail')}</span></p><p style="text-align:center; margin: 5px 0;">✖</p><p style="margin-bottom: 5px;"><strong style="color: #4da6ff;">{txt('val_mult')}</strong><br><span style="color: #aaa; font-size: 12px;">{txt('val_detail')}</span></p><hr style="margin: 10px 0; border-color: #555;"><p style="margin-bottom: 0;"><strong style="color: #4da6ff;">{txt('final_score')}</strong></p></div>""", unsafe_allow_html=True)

# --- MOBILE SEARCH ---
with st.expander(f"📱 {txt('analyze_mobile_btn')}", expanded=False):
    with st.form(key='mobile_form'):
        m_col1, m_col2 = st.columns([1, 1])
        with m_col1: m_market = st.selectbox(txt('market_label'), ["US", "Canada (TSX)", "HK (HKEX)"], key='m_m')
        with m_col2: m_ticker = st.text_input(txt('ticker_label'), value="NVDA", key='m_t').upper()
        m_submit = st.form_submit_button(txt('analyze_mobile_btn'), type="primary")

run_analysis = False
if d_submit:
    st.session_state.layout_mode = 'desktop'; st.session_state.active_ticker = d_ticker; st.session_state.active_market = d_market; run_analysis = True
elif m_submit:
    st.session_state.layout_mode = 'mobile'; st.session_state.active_ticker = m_ticker; st.session_state.active_market = m_market; run_analysis = True

# --- MAIN EXECUTION ---
if run_analysis:
    raw_t = st.session_state.active_ticker; mkt = st.session_state.active_market; final_t = raw_t
    if mkt == "Canada (TSX)" and ".TO" not in raw_t: final_t += ".TO"
    elif mkt == "HK (HKEX)": nums = ''.join(filter(str.isdigit, raw_t)); final_t = f"{nums.zfill(4)}.HK" if nums else f"{raw_t}.HK"

    with st.spinner(f"{txt('loading_data')} {final_t}..."):
        data = get_stock_data(final_t)

    if data:
        st.header(f"{data['name']} ({final_t})")
        st.caption(f"{txt('industry')}: {data['industry']} | {txt('currency')}: {data['currency']}")
        
        tab_fund, tab_tech, tab_fin, tab_news = st.tabs([txt('tab_value'), txt('tab_tech'), txt('tab_fin'), txt('tab_news')])

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
                    if match: s_str = match.group(1); s = float(s_str); r = res.replace(s_str, "").replace("|", "").replace("SCORE", "").replace("REASON", "").strip().strip(' :-=\n')
                    else: s, r = 0.0, res 
                    total_qual += s
                    qual_results.append((t_display, s, r))
                    with st.container(border=True):
                        c1, c2 = st.columns([4, 1])
                        with c1: st.markdown(f"**{t_display}**")
                        with c2: st.markdown(f"<h4 style='margin:0; text-align:right; color:#4da6ff;'>{s} <span style='font-size:14px; color:#888;'>/ 4</span></h4>", unsafe_allow_html=True)
                        st.progress(min(s/4.0, 1.0))
                        st.markdown(f"<div style='color:#666; font-size:14px; margin-top:5px;'>{r}</div>", unsafe_allow_html=True)
                prog_bar.empty(); status_text.empty()

            pe = data['pe']; min_pe = data['min_pe']; max_pe = data['max_pe']; position_pct = 0.0; mult = 1.0; color_code = "#FF4500"
            if pe and pe > 0 and max_pe > min_pe:
                position_pct = (pe - min_pe) / (max_pe - min_pe)
                if position_pct < 0.25: mult = 5.0
                elif position_pct < 0.50: mult = 4.0
                elif position_pct < 0.75: mult = 3.0
                elif position_pct < 1.00: mult = 2.0
                else: mult = 1.0
                mult = float(mult)
                if mult >= 4: color_code = "#00C805"
                elif mult >= 3: color_code = "#90EE90"
                elif mult >= 2: color_code = "#FFA500"
                else: color_code = "#FF4500"
            else: mult = 1.0; position_pct = 1.0; color_code = "#FF4500"
            final_score = round(total_qual * mult, 1) 
            verdict_text = txt('grade_avoid'); verdict_color = "#ffcccc"
            if final_score >= 75: verdict_text, verdict_color = txt('grade_strong_buy'), "#e6ffe6"
            elif final_score >= 60: verdict_text, verdict_color = txt('grade_buy'), "#f0fff0"
            elif final_score >= 45: verdict_text, verdict_color = txt('grade_hold'), "#fffff0"
            elif final_score >= 30: verdict_text, verdict_color = txt('grade_sell'), "#fff5e6"
            verdict_border_color = "#006600" if final_score >= 75 else "#b3b300" if final_score >= 45 else "#cc0000"

            with col_v:
                st.subheader(txt('quant_val_header'))
                with st.container(border=True):
                    st.caption(f"{txt('price')} ({data['currency']})"); st.metric("Price", f"{data['price']:.2f}", label_visibility="collapsed")
                    st.caption(txt('pe_ttm')); st.metric("Trailing PE", fmt_num(data['raw_info'].get('trailingPE')), label_visibility="collapsed")
                    st.caption(txt('pe_ratio')); st.metric("Forward PE", f"{pe:.2f}" if pe and pe > 0 else "N/A", label_visibility="collapsed")
                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.caption(txt('hist_low_pe')); c1.text_input("Low", value=f"{min_pe:.1f}", disabled=True, label_visibility="collapsed")
                    c2.caption(txt('hist_high_pe')); c2.text_input("High", value=f"{max_pe:.1f}", disabled=True, label_visibility="collapsed")
                    st.caption(txt('pe_pos')); safe_pct = max(0.0, min(1.0, position_pct)); st.progress(safe_pct)
                    cc1, cc2 = st.columns([1,1]); cc1.markdown(f"<small>{txt('pe_pos_low')}</small>", unsafe_allow_html=True); cc2.markdown(f"<div style='text-align:right'><small>{txt('pe_pos_high')}</small></div>", unsafe_allow_html=True)
                    st.divider(); st.subheader(txt('multiplier_label')); st.markdown(f"""<div class="multiplier-box" style="border: 2px solid {color_code}; color: {color_code};">x{mult:.0f}</div>""", unsafe_allow_html=True)

                    # --- MULTIPLIER EXPLANATION DROPDOWN ---
                    with st.expander(txt('mult_how')):
                        st.markdown(f"""
                        **{txt('mult_exp_title')}**  
                        {txt('mult_exp_desc')}
                        
                        **{txt('mult_formula')}**  
                        `({pe:.2f} - {min_pe:.2f}) / ({max_pe:.2f} - {min_pe:.2f}) = {position_pct*100:.1f}%`
                        
                        | {txt('mult_table_pos')} | {txt('mult_table_mult')} | {txt('mult_table_mean')} |
                        | :--- | :---: | :--- |
                        | 0% - 25% | **x5** | {txt('status_under')} |
                        | 25% - 50% | **x4** | {txt('status_fair')} |
                        | 50% - 75% | **x3** | {txt('status_fair')} |
                        | 75% - 100% | **x2** | {txt('status_over')} |
                        | > 100% | **x1** | {txt('status_over')} |
                        """)

            st.markdown(f"""<div class="final-score-box" style="border-color: {verdict_border_color}; padding: 25px;"><h3 style="color:#555; margin:0 0 20px 0; font-size: 22px;">{txt('score_calc_title')}</h3><div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 15px;"><div style="background: #f8f9fa; border-radius: 12px; padding: 15px; text-align: center; min-width: 120px; border: 1px solid #e0e0e0;"><div style="font-size: 13px; color: #666; margin-bottom: 5px;">{txt('calc_qual')}</div><div style="font-size: 32px; font-weight: 800; color: #333;">{total_qual:g}</div><div style="font-size: 12px; color: #999;">/ 20</div></div><div style="font-size: 24px; color: #bbb; font-weight: bold;">✖</div><div style="background: #f8f9fa; border-radius: 12px; padding: 15px; text-align: center; min-width: 120px; border: 1px solid #e0e0e0;"><div style="font-size: 13px; color: #666; margin-bottom: 5px;">{txt('calc_mult')}</div><div style="font-size: 32px; font-weight: 800; color: #333;">{mult:g}</div><div style="font-size: 12px; color: #999;">x1 - x5</div></div><div style="font-size: 24px; color: #bbb; font-weight: bold;">=</div><div style="background: #ffffff; border-radius: 12px; padding: 20px; text-align: center; min-width: 140px; border: 3px solid {verdict_border_color}; box-shadow: 0 4px 15px rgba(0,0,0,0.05);"><div style="font-size: 14px; color: #666; margin-bottom: 5px; font-weight: bold;">{txt('calc_result')}</div><div style="font-size: 48px; font-weight: 900; color: {verdict_border_color}; line-height: 1;">{final_score}</div></div></div><div style="margin-top: 25px;"><span style="background-color:{verdict_color}; padding: 8px 20px; border-radius: 20px; font-size: 16px; font-weight:bold; color:#333; border: 1px solid rgba(0,0,0,0.1);">{verdict_text}</span></div></div>""", unsafe_allow_html=True)
            with st.expander(txt('grading_scale'), expanded=False):
                st.markdown(f"""<table class="grade-table"><tr class="grade-green"><td>75 - 100</td><td>{txt('grade_strong_buy')}</td></tr><tr class="grade-lightgreen"><td>60 - 74.9</td><td>{txt('grade_buy')}</td></tr><tr class="grade-yellow"><td>45 - 59.9</td><td>{txt('grade_hold')}</td></tr><tr class="grade-orange"><td>30 - 44.9</td><td>{txt('grade_sell')}</td></tr><tr class="grade-red"><td>< 30</td><td>{txt('grade_avoid')}</td></tr></table>""", unsafe_allow_html=True)

        # ==========================================
        # TAB 2: TECHNICAL ANALYSIS
        # ==========================================
        with tab_tech:
            tech = calculate_technicals(data['history'])
            if tech:
                action_key = "act_avoid"; reason_key = "neutral"
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
                tc2.metric(txt('lbl_rsi'), f"{tech['rsi']:.1f}", delta=txt('status_high') if tech['rsi']>70 else txt('status_low') if tech['rsi']<30 else txt('status_ok'), delta_color="inverse")
                tc3.metric(txt('lbl_vol'), f"{tech['vol_ratio']:.2f}x"); tc4.metric(txt('squeeze'), "YES" if tech['is_squeezing'] else "No")
                c_sup, c_res = st.columns(2); c_sup.success(f"🛡️ {txt('support')}: {tech['support']:.2f}"); c_res.error(f"🚧 {txt('resistance')}: {tech['resistance']:.2f}")
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
            st.markdown("---"); st.subheader(txt('recent_div'))
            divs = data.get('dividends')
            if divs is not None and not divs.empty:
                divs_sorted = divs.sort_index(ascending=False).head(10); df_divs = divs_sorted.reset_index(); df_divs.columns = ["Date", "Amount"]
                df_divs['Date'] = df_divs['Date'].dt.strftime('%Y-%m-%d'); df_divs['Amount'] = df_divs['Amount'].apply(lambda x: f"{data['currency']} {x:.4f}")
                st.table(df_divs)
            else: st.info(txt('no_div'))
            st.caption(f"{txt('fiscal_year')}: {fmt_date(i.get('lastFiscalYearEnd'))}")

        # ==========================================
        # TAB 4: NEWS & EARNINGS
        # ==========================================
        with tab_news:
            st.subheader(txt('earn_title'))
            latest_earnings = None; earn_date = "N/A"
            if data['earnings_dates'] is not None and not data['earnings_dates'].empty:
                now = pd.Timestamp.now(tz=data['earnings_dates'].index.tz)
                past_earnings = data['earnings_dates'][data['earnings_dates'].index < now]
                if not past_earnings.empty:
                    latest_earnings = past_earnings.iloc[0]; earn_date = past_earnings.index[0].strftime('%Y-%m-%d')
            
            # Earnings Card
            est_eps = 0; act_eps = 0
            if latest_earnings is not None:
                with st.container(border=True):
                    ec1, ec2, ec3, ec4 = st.columns(4)
                    ec1.metric(txt('earn_date'), earn_date)
                    est_eps = latest_earnings.get('EPS Estimate')
                    ec2.metric(txt('earn_est_eps'), f"{est_eps:.2f}" if pd.notna(est_eps) else "-")
                    act_eps = latest_earnings.get('Reported EPS')
                    ec3.metric(txt('earn_act_eps'), f"{act_eps:.2f}" if pd.notna(act_eps) else "-")
                    
                    # FIX: Manual Surprise Calculation
                    surprise_str = "-"
                    surprise_val = 0
                    if pd.notna(est_eps) and pd.notna(act_eps) and est_eps != 0:
                        surprise_val = (act_eps - est_eps) / abs(est_eps)
                        surprise_str = f"{surprise_val*100:.2f}%"
                    else:
                         raw_surp = latest_earnings.get('Surprise(%)')
                         if pd.notna(raw_surp):
                             surprise_str = f"{raw_surp:.2f}%" if abs(raw_surp) > 5 else f"{raw_surp*100:.2f}%"

                    ec4.metric(txt('earn_surprise'), surprise_str, delta="Positive" if surprise_val > 0 else "Negative" if surprise_val < 0 else None)

            else: st.info("No specific earnings calendar data found.")

            st.markdown("---")
            
            # --- Q/Q Financial Trends Section ---
            st.subheader(txt('qq_title'))
            
            q_stmt = data['quarterly_financials']
            if q_stmt is not None and not q_stmt.empty and q_stmt.shape[1] >= 2:
                curr = q_stmt.iloc[:, 0] # Latest Quarter
                prev = q_stmt.iloc[:, 1] # Previous Quarter
                
                def calc_pct(cur, pre):
                    try: return ((cur - pre) / abs(pre)) * 100 if pre != 0 else None
                    except: return None
                
                def show_qq(label, cur_val, prev_val, is_curr=True, is_pct=False):
                    pct = calc_pct(cur_val, prev_val)
                    display_val = fmt_num(cur_val, is_currency=is_curr, is_pct=is_pct)
                    if is_pct: display_val = f"{cur_val*100:.2f}%" if pd.notna(cur_val) else "-"
                    st.metric(label, display_val, f"{pct:.2f}%" if pct is not None else "-", delta_color="normal")

                c_q1, c_q2, c_q3 = st.columns(3)
                with c_q1:
                    show_qq(txt('qq_rev'), curr.get('Total Revenue'), prev.get('Total Revenue'))
                    show_qq(txt('qq_op_inc'), curr.get('Operating Income'), prev.get('Operating Income'))
                with c_q2:
                    show_qq(txt('qq_net_inc'), curr.get('Net Income'), prev.get('Net Income'))
                    show_qq(txt('qq_op_exp'), curr.get('Operating Expense'), prev.get('Operating Expense'))
                with c_q3:
                    show_qq(txt('qq_eps'), curr.get('Basic EPS'), prev.get('Basic EPS'), is_curr=False)
                    try:
                        gm_c = curr.get('Gross Profit') / curr.get('Total Revenue')
                        gm_p = prev.get('Gross Profit') / prev.get('Total Revenue')
                        diff_bps = (gm_c - gm_p) * 100
                        st.metric(txt('qq_gross_marg'), f"{gm_c*100:.2f}%", f"{diff_bps:.2f} bps")
                    except: st.metric(txt('qq_gross_marg'), "-")
            else: st.info("Insufficient quarterly data for Q/Q comparison.")

            st.markdown("---")
            st.subheader(txt('ai_summary_title'))
            
            # Prepare context safely
            q_rev_disp = "N/A"
            if q_stmt is not None and not q_stmt.empty:
                q_rev_disp = fmt_num(q_stmt.iloc[:, 0].get('Total Revenue'), is_currency=True)

            news_text = ""
            if data['news']:
                for n in data['news'][:5]:
                    news_text += f"- {n.get('title', 'No Title')}\n"
            
            earn_context = f"Last Earnings Date: {earn_date}. Reported EPS: {act_eps if pd.notna(act_eps) else 'N/A'}. Revenue: {q_rev_disp}."
            full_context = f"{earn_context}\nRecent Headlines:\n{news_text}"
            
            with st.spinner(txt('loading_ai')):
                summary_text, _ = analyze_qualitative(data['name'], full_context, "EarningsSummary")
                st.success(summary_text)

            st.markdown("---")
            st.write("### 🔗 Official Sources")
            query = f"{data['name']} {final_t} Investor Relations Earnings Release"
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            st.link_button(txt('source_link'), search_url)

    else:
        st.error(f"Ticker '{final_t}' not found.")
