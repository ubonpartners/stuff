"""Tabulate and print structured results with optional section grouping and value colorization."""
import math
from collections import defaultdict
import numpy as np

def colour(s, r, g, b, bold=False):
    if bold:
        set_color=f"\033[1;38;2;{r};{g};{b}m"
    else:
        set_color=f"\033[38;2;{r};{g};{b}m"
    restore="\033[0m"
    return set_color+s+restore

def colorize(val, minval, maxval):
    if not isinstance(val, (int, float)) or math.isnan(val):
        return str(val)

    if val<0.01:
        return " "

    thr=0.02
    minval=min(minval, maxval-2*thr)
    delta_high = maxval - val
    delta_low = val - minval

    r=255
    g=255
    b=255
    bold=False
    if delta_high<thr:
        v=int((delta_high *255)/thr)
        r=v
        b=v
        bold=delta_high==0
    elif delta_low<thr:
        v=int((delta_low*255)/thr)
        g=v
        b=v
        bold=delta_low==0
    return colour(f"{val}", r, g, b, bold=bold)


def show_data(results_in, columns, column_text, sort_fn,
              section_key="dataset",
              add_section_dividers=True):
    #console_width = shutil.get_terminal_size().columns

    if sort_fn:
        results = sorted(results_in, key=sort_fn, reverse=True)
    else:
        results = results_in

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Please install pandas: pip install pandas")
    df = pd.DataFrame(results, columns=columns)

    def round_sf(x, sig=3):
        if isinstance(x, (int, float, np.floating)) and not (np.isnan(x) or np.isinf(x)) and x != 0:
            if (abs(x)<1):
                rounded=round(x,3)
            else:
                rounded = round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)
            return int(rounded) if (abs(rounded - int(rounded)) < 0.001) else rounded
        return x  # Leave NaN, inf, and non-numerics unchanged

    df_rounded = df.apply(lambda col: col.map(lambda x: round_sf(x, sig=3)))
    df_rounded = df_rounded.fillna('')

    data=df_rounded.to_dict(orient='records')
    # Step 1: Group by section_key
    grouped = defaultdict(list)
    for row in data:
        if section_key is None or not section_key in row:
            grouped["No Section"].append(row)
        else:
            grouped[row[section_key]].append(row)

    output_rows = []
    for section, rows in grouped.items():
        # Step 2: Find max/min per column (only for numeric columns)
        col_values = {col: [r.get(col, None) for r in rows] for col in columns}

        col_max = {}
        col_min = {}
        for col in columns:
            values = [v for v in col_values[col] if isinstance(v, (int, float)) and not math.isnan(v) and not v==0]
            if values:
                col_max[col] = max(values)
                col_min[col] = min(values)

        # Step 3: Color values
        for row in rows:
            display_row = []
            for col in columns:
                val = row.get(col, "")
                if (not col in col_min) or (not col in col_max):
                    if val==0:
                        val=" "
                    display_row.append(val)
                    continue
                if isinstance(val, (int, float)) and not math.isnan(val):
                    val = colorize(val, col_min[col], col_max[col])
                display_row.append(val)
            output_rows.append(display_row)

        # Optional: add separator line or empty line between sections
        if add_section_dividers and len(rows)>1:
            output_rows.append(['-']+[''] * (len(columns)-1))  # Empty line

    tablefmt = "minpadding"

    try:
        import tabulate
    except ImportError:
        raise ImportError("Please install tabulate: pip install tabulate")

    tabulate._table_formats[tablefmt] = tabulate.TableFormat(
        lineabove=tabulate.Line("", "-", " ", ""),
        linebelowheader=tabulate.Line("", "-", " ", ""),
        linebetweenrows=None,
        linebelow=tabulate.Line("", "-", " ", ""),
        headerrow=tabulate.DataRow("", " ", ""),
        datarow=tabulate.DataRow("", " ", ""),
        padding=0,
        with_header_hide=["lineabove", "linebelow"],
    )
    tabulate.multiline_formats[tablefmt] = tablefmt
    tabulate.tabulate_formats = list(sorted(tabulate._table_formats.keys()))

    table_str=tabulate.tabulate(output_rows, headers=column_text, tablefmt=tablefmt)
    #width = max(len(line) for line in table_str.splitlines())
    print("\n"+table_str)
    return table_str
