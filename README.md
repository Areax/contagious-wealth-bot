# Contagious Wealth Bot

Finds stocks worth selling **cash-secured puts** on and shows what each put pays. Runs on your own computer, opens in your browser.

**It never buys or sells anything.** You place any trade yourself in your own brokerage.

> **Not financial advice.** Educational screening tool only. Data is Yahoo Finance's free feed and can be delayed or wrong. Always check the contract in your broker before acting.

## Setup (one time, about 10 minutes)

### 1. Install Python

Go to <https://www.python.org/downloads/> and click the big yellow **Download Python** button. Run the installer.

> **Windows:** on the first installer screen, tick **"Add python.exe to PATH"** before clicking Install.

### 2. Download this project

On this GitHub page, click the green **Code** button, then **Download ZIP**.

Unzip it. You get a folder called `contagious-wealth-bot-main`. Move it somewhere easy to find, like your Desktop.

### 3. Open a terminal inside that folder

- **Mac:** right-click the `contagious-wealth-bot-main` folder, then choose **New Terminal at Folder**. (If you don't see it: open the **Terminal** app, type `cd ` with a space, drag the folder into the window, press Enter.)
- **Windows:** open the `contagious-wealth-bot-main` folder, click the address bar at the top, type `cmd`, press Enter.

### 4. Install the packages

Copy this line into the terminal and press Enter. Lots of scrolling text is normal. Wait until it stops.

**Mac:**

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Windows:**

```bat
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

## Run it

In the same terminal:

**Mac:**

```bash
source .venv/bin/activate && python main.py web
```

**Windows:**

```bat
.venv\Scripts\activate && python main.py web
```

Then open <http://127.0.0.1:8899> in your browser.

The first scan takes a minute or two. Stocks appear in the table as they are ready. Hover any **?** in the page for an explanation.

To stop: go back to the terminal and press **Ctrl+C**.

## Running it again later

Skip Setup. Do step 3 (open a terminal in the folder), then the **Run it** line for your system.

## If something goes wrong

- **`python` is not recognized / command not found.** Python is not installed, or on Windows the PATH box was not ticked. Reinstall Python, then open a new terminal.
- **`No module named 'flask'`.** Step 4 did not finish. Run it again.
- **`Address already in use`.** Something else is on port 8899. Run `python main.py web --port 8900` and open <http://127.0.0.1:8900>.
- **Many rows show `-`.** Normal outside US market hours (9:30am to 4pm Eastern, Mon to Fri). Yahoo has no live quotes then. Try again when the market is open.
- **Yahoo says rate limited.** Wait a few minutes and scan a smaller list.

Everything is stored locally in the `cache/` folder. Nothing about you is sent anywhere. Deleting `cache/` resets the tool safely.

More detail (command line, how it is built, known gaps): [docs/REFERENCE.md](docs/REFERENCE.md).
