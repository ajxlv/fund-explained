"""Plain-English, one-sentence explanations of the terms used in the app."""

GLOSSARY: dict[str, str] = {
    "NAV": "Net Asset Value: the price of one unit of the fund, published every trading day.",
    "SIP": "Systematic Investment Plan: investing a fixed amount every month instead of all at "
           "once.",
    "CAGR": "Compound Annual Growth Rate: the steady yearly growth that would turn the starting "
            "value into the ending value.",
    "Worst fall (max drawdown)": "The biggest drop from a high point to a later low point, "
                                 "before the fund climbed back.",
    "Recovery time": "How long the fund took to climb from its lowest point back to the "
                     "previous high.",
    "Volatility": "How much the fund's value jumps around day to day, scaled to a year; higher "
                  "means a bumpier ride.",
    "Downside volatility": "Like volatility, but counting only the days the fund fell.",
    "Beta": "How much the fund tends to move when the Nifty moves 1%; above 1 means it swings "
            "more than the market.",
    "Correlation": "How closely two funds' daily ups and downs line up, from -1 (opposite) to "
                   "1 (in lockstep).",
    "Tracking error": "How much the fund's daily returns differ from the Nifty's, scaled to a "
                      "year.",
    "Nifty 50": "An index of 50 of the largest companies on India's National Stock Exchange, "
                "often used as \"the market\".",
    "Index fund": "A fund that simply copies an index such as the Nifty 50 instead of picking "
                  "stocks.",
    "Direct plan": "The version of a fund bought straight from the fund house, with no "
                   "distributor commission, so it has lower fees.",
    "Growth option": "Profits stay invested in the fund instead of being paid out, so the NAV "
                     "includes them.",
    "Fund house (AMC)": "Asset Management Company: the firm that runs the fund.",
    "Category": "The SEBI label for what a fund is allowed to hold, such as Large Cap (mostly "
                "the 100 biggest companies).",
    "Lookalike": "A fund whose daily returns moved very similarly to this one over the last 3 "
                 "years.",
    "Cluster": "A group of funds that the computer found behave alike, ignoring their official "
               "labels.",
    "Risk label": "Low / Moderate / High / Very High, based on how this fund's volatility ranks "
                  "against the other funds here.",
}
