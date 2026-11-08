"""Build the one-file report in the order required by the official brief."""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from pypdf import PdfReader
from PIL import Image as PillowImage

from ml_core import ROOT
from build_readme import build, github_url


class EvidenceImage(Image):
    """Keep the source capture intact and optionally cover a pointer in empty UI chrome."""

    def __init__(self, filename, width, height, cursor_box=None, background_column=None):
        super().__init__(filename, width=width, height=height)
        self.pointer_rows = []
        if cursor_box is None:
            return
        left, top, box_width, box_height = cursor_box
        with PillowImage.open(filename) as source:
            source = source.convert("RGB")
            source_width, source_height = source.size
            if not (0 <= left < left + box_width <= source_width
                    and 0 <= top < top + box_height <= source_height
                    and background_column is not None
                    and 0 <= background_column < source_width):
                raise ValueError("Cursor box and reference column must be inside the screenshot.")
            self.pointer_scale = width / source_width
            self.pointer_left, self.pointer_width = left, box_width
            # Sample an unobstructed column of the same empty horizontal toolbar.
            for row in range(top, top + box_height):
                color = source.getpixel((background_column, row))
                if self.pointer_rows and self.pointer_rows[-1][2] == color:
                    self.pointer_rows[-1][1] += 1
                else:
                    self.pointer_rows.append([row, 1, color])

    def draw(self):
        super().draw()
        if not self.pointer_rows:
            return
        self.canv.saveState()
        scale = self.pointer_scale
        for row, count, color in self.pointer_rows:
            self.canv.setFillColorRGB(*(component / 255 for component in color))
            self.canv.rect(self.pointer_left * scale,
                           self.drawHeight - (row + count) * scale,
                           self.pointer_width * scale, count * scale + 0.001,
                           stroke=0, fill=1)
        self.canv.restoreState()


def inline(text):
    safe=html.escape(text)
    safe=re.sub(r"\[([^\]]+)\]\((https?://[^ )]+)\)",r'<link href="\2" color="black"><u>\1</u></link>',safe)
    safe=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",safe)
    return safe


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--draft",action="store_true")
    parser.add_argument("--github-url")
    parser.add_argument("--app-url")
    parser.add_argument("--lab-screenshot",type=Path)
    parser.add_argument("--cursor-box", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Optional pointer-only box in source pixels, within empty horizontal UI chrome.")
    parser.add_argument("--cursor-background-column", type=int,
                        help="Unobstructed background column used for the PDF pointer cover.")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if not args.draft:
        if not all([args.github_url,args.app_url,args.lab_screenshot]):
            parser.error("Final PDF needs --github-url, --app-url and --lab-screenshot. Use --draft for review.")
        github_url(args.github_url)
        parsed=urlparse(args.app_url)
        if parsed.scheme!="https" or not (parsed.hostname or "").endswith(".streamlit.app"):
            parser.error("Provide the actual HTTPS Streamlit Community Cloud app URL.")
        if not args.lab_screenshot.is_file():
            parser.error("The genuine BITS Lab screenshot file does not exist.")
        execution=json.loads((ROOT/"results/execution_summary.json").read_text())
        if execution["execution_context_operator_declared"]!="bits-lab":
            parser.error("Run train.py --execution-context bits-lab in the actual BITS lab first, then copy its results back.")
    readme=build(args.github_url,args.app_url)
    output=args.output or ROOT/"output/pdf"/("2026AJ05692_ML_Assignment2_REVIEW.pdf" if args.draft else "2026AJ05692_ML_Assignment2.pdf")
    output.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyAcademic",fontName="Helvetica",fontSize=9.5,leading=13,spaceAfter=7))
    styles.add(ParagraphStyle(name="TableAcademic",fontName="Helvetica",fontSize=8,leading=10))
    styles.add(ParagraphStyle(name="TitleAcademic",fontName="Helvetica-Bold",fontSize=15,leading=19,alignment=TA_CENTER,spaceAfter=12))
    body=styles["BodyAcademic"]
    story=[Paragraph("Machine Learning - Assignment 2",styles["TitleAcademic"]),
           Paragraph("DINESH BEHERA | Student | BITS ID: 2026AJ05692 | AIML ZG565",body)]
    if args.draft:
        story.append(Paragraph("<b>LOCAL REVIEW COPY - NOT READY TO SUBMIT</b>",body))
        story.append(Paragraph("GitHub publication, Streamlit deployment and genuine BITS Virtual Lab execution evidence are pending. This copy records local work only.",body))
    def section(title,text):
        story.append(Paragraph(title,styles["Heading2"]))
        story.append(Paragraph(inline(text),body))
    section("1. GitHub Repository Link",args.github_url or "Pending student publication.")
    section("2. Live Streamlit App Link",args.app_url or "Pending student deployment on Streamlit Community Cloud.")
    if args.github_url:
        story[-3]=Paragraph(f'<link href="{html.escape(args.github_url)}"><u>{html.escape(args.github_url)}</u></link>',body)
    if args.app_url:
        story[-1]=Paragraph(f'<link href="{html.escape(args.app_url)}"><u>{html.escape(args.app_url)}</u></link>',body)
    story.append(Paragraph("3. BITS Virtual Lab execution screenshot",styles["Heading2"]))
    if args.lab_screenshot:
        width,height=ImageReader(str(args.lab_screenshot)).getSize()
        scale=min(480/width,380/height)
        story.append(EvidenceImage(str(args.lab_screenshot), width=width*scale, height=height*scale,
                                   cursor_box=args.cursor_box,
                                   background_column=args.cursor_background_column))
    else:
        story.append(Paragraph("Pending. Local execution output is not BITS Virtual Lab proof. Insert one genuine screenshot of assignment execution in the lab.",body))
    story.append(PageBreak())
    story.append(Paragraph("4. GitHub README content",styles["Heading1"]))
    lines=readme.read_text(encoding="utf-8").splitlines()
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                raw=[x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?",x) for x in raw):
                    rows.append([Paragraph(inline(x),styles["TableAcademic"]) for x in raw])
                i+=1
            widths=[111]+[61.3]*6 if len(rows[0])==7 else [110,369]
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign="LEFT",splitByRow=0)
            table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.black),
                                       ("VALIGN",(0,0),(-1,-1),"TOP"),
                                       ("TOPPADDING",(0,0),(-1,-1),6),
                                       ("BOTTOMPADDING",(0,0),(-1,-1),6)]))
            story.extend([table,Spacer(1,10)])
            continue
        if line.startswith("#"):
            if line == "### Comparison table":
                story.append(PageBreak())
            n=len(line)-len(line.lstrip("#"))
            style=styles["Heading1" if n==1 else "Heading2" if n==2 else "Heading3"]
            story.append(Paragraph(inline(line.lstrip("#").strip()),style))
        else:
            paragraph=Paragraph(inline(line),body)
            if line.startswith("All entries are measured"):
                paragraph.keepWithNext=True
            story.append(paragraph)
        i+=1
    def footer(canvas,doc):
        canvas.saveState()
        canvas.setFont("Helvetica",8)
        canvas.drawString(42,25,"2026AJ05692 | ML Assignment 2" + (" | REVIEW" if args.draft else ""))
        canvas.drawRightString(A4[0]-42,25,str(doc.page))
        canvas.restoreState()
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=42,leftMargin=42,
                          topMargin=38,bottomMargin=40,title="Machine Learning Assignment 2",
                          author="DINESH BEHERA")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(output)
    text="\n".join(page.extract_text() for page in reader.pages)
    for required in ["2026AJ05692","Problem statement","Dataset description","Models used","Accuracy","MCC"]:
        if required not in text:
            raise ValueError(f"Required PDF text missing: {required}")
    print(f"Created {output} | {len(reader.pages)} pages | SHA256 {hashlib.sha256(output.read_bytes()).hexdigest()}")
    if not args.draft:
        print("Verify both links in a signed-out browser and inspect the PDF before submitting.")


if __name__=="__main__":
    main()
