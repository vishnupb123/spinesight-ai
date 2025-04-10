# utils/pdf_generator.py
from flask import render_template
import pdfkit

def generate_individual_report(report_data: dict, pdfkit_config: dict) -> bytes:
    """
    Generate a PDF for an individual prediction report.
    """
    html = render_template('report.html', report_data=report_data)
    config = pdfkit.configuration(**pdfkit_config)
    pdf = pdfkit.from_string(html, False, configuration=config)
    return pdf

def generate_bulk_report(results: list, pdfkit_config: dict) -> bytes:
    """
    Generate a bulk report PDF for multiple predictions.
    """
    html = render_template('report_bulk.html', results=results)
    config = pdfkit.configuration(**pdfkit_config)
    pdf = pdfkit.from_string(html, False, configuration=config)
    return pdf
