"""Number formatting for display (Indian digit grouping)."""
import math

DASH = "—"


def fmt_inr(x: float | None, decimals: int = 0, symbol: bool = True) -> str:
    """Format rupees with Indian grouping: 123456 -> "₹1,23,456"."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return DASH
    sign = "-" if x < 0 else ""
    whole, _, frac = f"{abs(x):.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        groups.insert(0, head)
        whole = ",".join(groups) + "," + tail
    out = whole + (f".{frac}" if frac else "")
    return f"{sign}{'₹' if symbol else ''}{out}"


def fmt_pct(x: float | None, decimals: int = 1, signed: bool = False) -> str:
    """0.1234 -> "12.3%" (fractions in, percent out)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return DASH
    return f"{x * 100:{'+' if signed else ''}.{decimals}f}%"
