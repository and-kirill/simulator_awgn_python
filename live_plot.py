"""
Live plot server
"""
# This file is part of the simulator_awgn_python distribution
# https://github.com/and-kirill/simulator_awgn_python/.
# Copyright (c) 2023 Kirill Andreev.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import logging

from dash import dcc, html, Dash
from dash.dependencies import Input, Output

import plotly.graph_objs as go


class PlotServer:
    """
    Create/run Dash application server
    """
    def __init__(self, **kwargs):
        # Disable HTML log
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        self.ip_address = kwargs.get('ip_address', '127.0.0.1')
        self.port = kwargs.get('port', 8888)
        self.update_ms = kwargs.get('update_ms', 1000)
        self.title = kwargs.get('title')
        self.postproc = kwargs.get('postproc_instance')

    def styling(self, fig):
        """
        Figure styling: axes names, grid, ticks format, etc...
        """
        fig.update_layout(
            uirevision=True,  # Keep zoom and pan between updates
            autosize=True,
            yaxis={
                'title': 'Error rate',
                'showexponent': 'all',
                'dtick': 'D0',
                'exponentformat': 'power',
                'type': 'log',
                'minor': {'showgrid': True}
            },
            xaxis={
                'title': 'Signal to noise ratio, [dB]',
                'minor': {'showgrid': True}
            },
            template='plotly_dark',
            title=self.title
        )
        return fig

    def empty_fig(self):
        """
        Generate empty figure
        """
        fig = go.Figure()
        return self.styling(fig)

    def update_figure(self, _):
        """
        Client callback to update figure
        """
        fig = self.empty_fig()
        data = self.postproc.get()
        if data is None:
            return fig

        fig.add_trace(go.Scatter(
            x=data['snr'],
            y=data['fer'],
            error_y={
                'type': 'data',
                'symmetric': False,
                'array': data['fer_e_plus'],
                'arrayminus': data['fer_e_minus']
            },
            mode='markers',
            name='Simulated FER'
        ))
        fig.add_scatter(
            x=data['snr'],
            y=data['fer_fit'],
            name='FER Fit (Bernoulli loss)'
        )
        fig.add_scatter(
            x=data['snr'],
            y=data['in_ber'],
            mode='markers',
            name='Uncoded BER'
        )
        fig.add_scatter(
            x=data['snr'],
            y=data['in_ber_ref'],
            name='Uncoded BER (theory)'
        )
        fig.add_scatter(
            x=data['snr'],
            y=data['ber'],
            mode='markers',
            name='Simulated BER'
        )
        fig.add_scatter(
            x=data['snr'],
            y=data['ber_fit'],
            name='Output BER Fit (Bernoulli loss)'
        )
        return fig

    def run(self):
        """
        Run server
        """
        fig = self.empty_fig()
        app = Dash(__name__, update_title=None)  # remove "Updating..." from title
        app.layout = html.Div([
            dcc.Graph(
                id='graph',
                figure=fig,
                style={
                    'margin': 0,
                    'width': '100hh',
                    'height': '100vh'
                }),
            dcc.Interval(
                id="interval",
                interval=self.update_ms,
                n_intervals=0  # Unlimited number of updates
            )
        ])
        app.callback(
            Output('graph', 'figure'),
            [Input('interval', 'n_intervals')],
        )(self.update_figure)
        app.run_server(host=self.ip_address, port=self.port, debug=False)
