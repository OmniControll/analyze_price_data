from flask import Flask, render_template, request
from stock_analysis import analyze_stocks

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        tickers = request.form.get('tickers')
        tickers = tickers.split(',')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        optimized_weights, top_portfolios, monte_carlo_results, plot_path = analyze_stocks(tickers, start_date, end_date)

        return render_template('results.html', optimized_weights=optimized_weights, top_portfolios=top_portfolios, monte_carlo_results=monte_carlo_results, plot_path=plot_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
