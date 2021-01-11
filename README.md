# AlgoTrading
------------

This project implements the necessary functionalities to engage in crypto algorithmic trading.


## Features

- [x] **Written in Python 3.8+**
- [x] **Persistence**: Persistence is achieved through sqlite.
- [x] **Paper-Trading**: Run the bot without playing money.
- [x] **Backtesting**: Run a simulation of a buy/sell strategy on a specific pair.
- [x] **Strategy templates**: The project comes with build-in strategies that the user can fine-tune.
- [x] **Strategy Optimization by GridSearch**: Use a GridSearch to optimize the buy/sell strategy parameters with real exchange data.
- [x] **Dashboard GUI**: Monitor the performances of the bot through an elegant Dash GUI.
- [x] **Telegram Interface**: Manage the bot with Telegram. Currently Read-only.
- [x] **Performance status report**: Provide a performance status of your current trades.


## Basic Usage

Store your Binance API key in a 'credentials.txt' in /assets

### Structure

The code is organized as follows :
- `Trading.py`: Manages the 24/7 trading process.
- `Dashboard.py`: GUI to have metrics on the algo's behavior.
- `Telegram_interface.py`: run the file and send `/start` to the telegram bot from telegram.
- `Database.py`: Manages the databases. One database for paper_trading, another for live trading.
- `Echange.py`: Manages the interface with Binance's REST API.
- `/assets.py`: stores some usefull files, and the Binance and Telegram credentials.
- `/backtesting.py`: Multiple scripts to backtest strategies..
- `/experimentations.py`: Files in progress to test personal ideas.



## Disclaimer

This software is for educational purposes only. Do not risk money which
you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHOR ASSUMES NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running the algo in paper-trading mode and do not engage money
before you understand how it works and what profit/loss you should
expect.