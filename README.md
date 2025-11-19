# Local-financial-AI-Analyst
this is a Local financial AI Analyst
Step A: Prepare LM Studio
1. Open LM Studio.
2. Go to the Chat tab and load your Qwen3-VL-8B model.
3. Click the Developer/Server icon (usually looks like <-> on the left sidebar).
4. Click Start Server.
5. Ensure the server URL is http://localhost:1234

Step B: Install Python Dependencies
You need to remove groq and install openai (which works with LM Studio). Open your terminal/command prompt:
code
Bash >>
pip install streamlit yfinance pandas numpy openai

4. How to Run It
Make sure LM Studio server is running (Green "Start Server" button clicked).
Open your command prompt in the folder where you saved local_app.py.
Run the command:
code
Bash
streamlit run local_app.py
