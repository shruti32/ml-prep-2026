import typer
import pandas as pd
from pathlib import Path
from .stats import compute_stats, compute_correlation
from .report import generate_report

app = typer.Typer()


@app.command()
def analyse(
    input_file: Path = typer.Argument(..., help="Path to input CSV file"),
    output_file: Path = typer.Option("report.html", help="Path for output HTML report"),
):
    """Analyse a CSV dataset and produce an HTML statistical report."""

    typer.echo(f"Loading {input_file}...")

    if not input_file.exists():
        typer.echo(f"Error: file {input_file} not found.", err=True)
        raise typer.Exit(1)

    df = pd.read_csv(input_file)
    typer.echo(f"Loaded {len(df)} rows x {len(df.columns)} columns")

    typer.echo("Computing statistics...")
    stats = compute_stats(df)
    correlation = compute_correlation(df)

    typer.echo("Generating report...")
    html = generate_report(
        stats=stats,
        correlation=correlation,
        filename=input_file.name,
        n_rows=len(df),
    )

    output_file.write_text(html)
    typer.echo(f"Report saved to {output_file}")


def main():
    app()


if __name__ == "__main__":
    main()
