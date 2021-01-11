import pandas as pd
results = dict()
results['length_fast']   = [2,	2,	2,	2,	2,	2,	4,	4,	2,	2,	2,	2,	2,	2,	4,	4,	4,	4,	4,	4,	6,	6,	6,	6,	4,	4,	4,	4,	6,	6,	6,	6,	6,	6,	8,	8,	8,	8,	8,	8,	6,	6,	8,	8,	8,	8,	8,	8,	10,	10,	10,	10,	10,	10,	10,	10,	10,	10,	10,	10,]
results['length_slow']   = [40,	40,	30,	20,	20,	30,	20,	20,	40,	40,	20,	30,	20,	30,	20,	20,	30,	40,	30,	40,	20,	30,	30,	20,	30,	40,	30,	40,	30,	20,	30,	20,	40,	40,	20,	30,	40,	30,	20,	40,	40,	40,	30,	30,	20,	40,	40,	20,	20,	20,	30,	30,	40,	40,	30,	20,	30,	20,	40,	40,]
results['stop_loss_pct'] = [2,	4,	2,	4,	2,	4,	2,	4,	3,	5,	5,	3,	3,	5,	3,	5,	2,	2,	4,	4,	2,	2,	4,	4,	3,	3,	5,	5,	3,	3,	5,	5,	2,	4,	2,	2,	2,	4,	4,	4,	3,	5,	5,	3,	3,	3,	5,	5,	2,	4,	2,	4,	2,	4,	3,	3,	5,	5,	5,	5,]
results['quote_profits'] = [13.9,	13.5,	5.0,	12.7,	0.0,	13.2,	3.0,	14.1,	13.3,	12.8,	13.4,	15.7,	9.3,	5.0,	0.2,	12.5,	5.5,	16.7,	12.8,	13.8,	2.1,	1.9,	11.6,	7.5,	12.2,	16.1,	7.3,	11.1,	13.0,	-3.4,	8.9,	8.7,	4.9,	16.8,	0.0,	11.1,	-0.7,	18.4,	5.4,	12.8,	18.1,	15.5,	15.3,	19.9,	-5.3,	11.3,	10.6,	2.7,	-17.6,	11.3,	10.0,	26.7,	10.2,	19.6,	23.4,	-8.2,	20.7,	1.4,	19.1,	19.1,]



if __name__ == "__main__":

    import plotly.graph_objs as go
    fig = go.Figure()

    df_ = pd.DataFrame({'length_fast'   : results['length_fast'],
                        'length_slow'   : results['length_slow'],
                        'stop_loss_pct' : results['stop_loss_pct'],
                        'quote_profits' : results['quote_profits']})

    fig.add_trace(go.Scatter3d(x=df_.loc[:,'length_fast'], y=df_.loc[:,'length_slow'], z=df_.loc[:,'stop_loss_pct'],
                               mode='markers',
                               marker=dict(
                                   size       = 5,
                                   color      = df_.loc[:,'quote_profits'],      	# set color to an array/list of desired values
                                   colorscale = 'Viridis',                   		# choose a colorscale
                                   opacity    = 0.8,
                                   colorbar   = dict(thickness = 20,
                                                     title     = "BTC profits %"),
                               )))
    fig.update_layout(scene = dict(xaxis_title='Length fast',
                                   yaxis_title='Length slow',
                                   zaxis_title='stop_loss_pct',))

    fig.update_layout(title = f"ETHBTC - Crossover SSF - 1m",)
    fig.show()


