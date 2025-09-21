from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import date

def build_weekly_report(data, out_file="weekly_report.pdf"):
    doc = SimpleDocTemplate(out_file, pagesize=A4)
    elements = []

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=22,
        textColor=colors.HexColor("#2E86C1"),
        spaceAfter=20
    )
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        textColor=colors.HexColor("#1A5276"),
        spaceBefore=12, spaceAfter=8
    )
    normal = styles['Normal']

    # Titre
    today = date.today().strftime("%d %B %Y")
    elements.append(Paragraph(f"📊 Rapport Hebdomadaire — {today}", title_style))
    elements.append(Spacer(1, 12))

    # Résumé
    elements.append(Paragraph("Résumé de la semaine", h2_style))
    elements.append(Paragraph(data.get("summary", "Aucun résumé disponible."), normal))
    elements.append(Spacer(1, 16))

    # Nouveaux projets
    if data.get("projects"):
        elements.append(Paragraph("🚀 Nouveaux Projets", h2_style))
        table_data = [["Entreprise", "Projet", "Domaine"]]
        for row in data["projects"]:
            table_data.append(row)
        table = Table(table_data, colWidths=[150, 200, 150])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#5DADE2")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER')
        ]))
        elements.append(table)
        elements.append(Spacer(1, 16))

    # Tendances / Graphique (placeholder)
    elements.append(Paragraph("📈 Tendances", h2_style))
    elements.append(Paragraph("Un graphique peut être généré avec matplotlib et inséré ici.", normal))

    # Build PDF
    doc.build(elements)
    print(f"✅ Rapport généré: {out_file}")

if __name__ == "__main__":
    data = {
        "summary": "Cette semaine, plusieurs initiatives IA ont émergé dans le secteur financier.",
        "projects": [
            ["BNP Paribas", "Chatbot interne RH", "IA"],
            ["LVMH", "Analyse prédictive ventes", "Retail"],
            ["Capgemini", "Outil de veille concurrentielle", "Consulting"]
        ]
    }
    build_weekly_report(data)
