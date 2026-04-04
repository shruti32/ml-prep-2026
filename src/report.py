from jinja2 import Template
from .stats import ColumnStats
import pandas as pd


TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Statistical Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
        h1   { color: #1F4E79; }
        h2   { color: #2E75B6; border-bottom: 2px solid #2E75B6; padding-bottom: 6px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; font-size: 14px; }
        th   { background: #2E75B6; color: white; padding: 10px; text-align: left; }
        td   { padding: 8px 10px; border-bottom: 1px solid #ddd; }
        tr:nth-child(even) { background: #f5f9ff; }
        .missing-high { color: #c00; font-weight: bold; }
        .skewed       { color: #e65c00; }
        .summary-box  { background: #f0f4ff; border-left: 4px solid #2E75B6;
                        padding: 12px 20px; margin-bottom: 20px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Statistical Analysis Report</h1>

    <div class="summary-box">
        <strong>Dataset:</strong> {{ filename }}<br>
        <strong>Rows:</strong> {{ n_rows }} &nbsp;|&nbsp;
        <strong>Columns:</strong> {{ n_cols }} &nbsp;|&nbsp;
        <strong>Numeric columns:</strong> {{ n_numeric }} &nbsp;|&nbsp;
        <strong>Categorical columns:</strong> {{ n_categorical }}
    </div>

    <h2>Missing Values</h2>
    <table>
        <tr><th>Column</th><th>Type</th><th>Missing</th><th>Missing %</th></tr>
        {% for s in stats %}
        <tr>
            <td>{{ s.name }}</td>
            <td>{{ s.dtype }}</td>
            <td>{{ s.missing }}</td>
            <td class="{{ 'missing-high' if s.missing_pct > 10 else '' }}">
                {{ s.missing_pct }}%
            </td>
        </tr>
        {% endfor %}
    </table>

    <h2>Descriptive Statistics (Numeric Columns)</h2>
    <table>
        <tr>
            <th>Column</th><th>Count</th><th>Mean</th><th>Median</th>
            <th>Std Dev</th><th>Min</th><th>Max</th>
            <th>Skewness</th><th>Kurtosis</th><th>Shape Assessment</th>
        </tr>
        {% for s in stats if s.mean is not none %}
        <tr>
            <td>{{ s.name }}</td>
            <td>{{ s.count }}</td>
            <td>{{ s.mean }}</td>
            <td>{{ s.median }}</td>
            <td>{{ s.std }}</td>
            <td>{{ s.min }}</td>
            <td>{{ s.max }}</td>
            <td class="{{ 'skewed' if s.skewness|abs > 1 else '' }}">{{ s.skewness }}</td>
            <td>{{ s.kurtosis }}</td>
            <td>{{ s.shape_assessment }}</td>
        </tr>
        {% endfor %}
    </table>

    {% if corr_html %}
    <h2>Correlation Matrix</h2>
    {{ corr_html }}
    {% endif %}

</body>
</html>
"""


def generate_report(
    stats: list[ColumnStats],
    correlation: pd.DataFrame,
    filename: str,
    n_rows: int,
) -> str:
    n_numeric = sum(1 for s in stats if s.mean is not None)
    n_categorical = sum(1 for s in stats if s.mean is None)
    corr_html = (
        correlation.to_html(classes="corr-table") if not correlation.empty else ""
    )

    template = Template(TEMPLATE)
    return template.render(
        stats=stats,
        corr_html=corr_html,
        filename=filename,
        n_rows=n_rows,
        n_cols=len(stats),
        n_numeric=n_numeric,
        n_categorical=n_categorical,
    )
