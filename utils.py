from fpdf import FPDF
import re

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Product Specifications Summary', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def create_pdf(text_content):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    # Simple markdown-like parsing to clearer text for PDF
    # Remove excessive markdown bolding syntax for cleaner PDF
    clean_text = text_content.replace('**', '')
    
    # Split text into lines to handle them
    lines = clean_text.split('\n')
    
    for line in lines:
        if line.strip().startswith('---'):
             pdf.ln(5)
             pdf.line(10, pdf.get_y(), 200, pdf.get_y())
             pdf.ln(5)
        else:
            # Handle encoding issues by identifying replacing non-latin-1 characters
            # or simply using a compatible font/encoding if needed. 
            # For simplicity in FPDF free version, we stick to latin-1 safe text 
            # or replace chars.
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, safe_line)
            
    return pdf.output(dest='S').encode('latin-1', 'replace')

def clean_html_content(soup):
    """
    Cleans up HTML content to remove scripts, styles, and other unnecessary tags
    before sending to LLM.
    """
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header", "meta", "noscript"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text
