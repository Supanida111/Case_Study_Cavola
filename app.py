"""
Calvora — Dash App Entry Point (Modern UI Version)
5-page dashboard: Home, Unsupervised, Supervised, Business Insight, Predict.
Features: Glassmorphism, Modern Typography, Smooth Interactions.
"""
import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from pages import home, unsupervised, supervised_page, business_insight, predict

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Calvora — ระบบวิเคราะห์การตลาดด้วย AI"
)

# ============ Modern Navbar ============
NAV = dbc.Navbar(
    dbc.Container([
        # Brand & Logo
        html.A(
            dbc.Row([
                dbc.Col(
                    html.Span(
                        "✨",
                        className="me-2 floating",
                        style={"fontSize": "1.8rem", "display": "inline-block"}
                    ),
                    width="auto"
                ),
                dbc.Col(
                    dbc.NavbarBrand(
                        "Calvora AI",
                        className="ms-2",
                        style={
                            "fontFamily": "'Mali', sans-serif",
                            "fontWeight": "700",
                            "fontSize": "1.6rem",
                            "letterSpacing": "-0.5px"
                        }
                    ),
                    width="auto"
                ),
            ], align="center", className="g-0"),
            href="/",
            style={"textDecoration": "none"},
        ),

        # Navbar Toggler
        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),

        # Navigation Links
        dbc.Collapse(
            dbc.Nav([
                dbc.NavItem(dbc.NavLink(
                    [html.I(className="bi bi-house-door me-2"), "หน้าแรก"],
                    href="/",
                    style={"transition": "all 0.3s ease"}
                )),
                dbc.NavItem(dbc.NavLink(
                    [html.I(className="bi bi-diagram-3 me-2"), "Unsupervised"],
                    href="/unsupervised",
                    style={"transition": "all 0.3s ease"}
                )),
                dbc.NavItem(dbc.NavLink(
                    [html.I(className="bi bi-cpu me-2"), "Supervised"],
                    href="/supervised",
                    style={"transition": "all 0.3s ease"}
                )),
                dbc.NavItem(dbc.NavLink(
                    [html.I(className="bi bi-lightbulb me-2"), "Business Insight"],
                    href="/insight",
                    style={"transition": "all 0.3s ease"}
                )),
                dbc.NavItem(dbc.NavLink(
                    [html.I(className="bi bi-magic me-2"), "Predict"],
                    href="/predict",
                    style={"transition": "all 0.3s ease"}
                )),
            ], className="ms-auto", navbar=True),
            id="navbar-collapse",
            navbar=True,
        ),
    ], fluid=True),
    color="light",
    dark=False,
    sticky="top",
    className="navbar",
    style={"marginBottom": "2rem"}
)

# ============ Main Layout ============
app.layout = html.Div([
    dcc.Location(id="url"),
    NAV,
    html.Div(
        id="page-content",
        style={
            "padding": "0 24px 40px 24px",
            "fontFamily": "'Quicksand', sans-serif",
            "minHeight": "calc(100vh - 100px)"
        }
    ),
], style={"backgroundColor": "transparent"})


# ============ Callbacks ============
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    """Toggle navbar collapse on mobile."""
    if n:
        return not is_open
    return is_open


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route(pathname):
    """Route to appropriate page based on URL pathname."""
    if pathname == "/unsupervised":
        return unsupervised.layout()
    if pathname == "/supervised":
        return supervised_page.layout()
    if pathname == "/insight":
        return business_insight.layout()
    if pathname == "/predict":
        return predict.layout()
    return home.layout()


if __name__ == "__main__":
    app.run(debug=True, port=8050)
