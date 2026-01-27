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



def create_cluster_pdf(cluster_data, x_label, y_label):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    title = 'Cluster Analysis Report'
    pdf.cell(0, 10, title, 0, 1, 'L')
    pdf.ln(5)
    
    # Sanitize labels for header
    safe_x_label = x_label.encode('latin-1', 'replace').decode('latin-1')
    safe_y_label = y_label.encode('latin-1', 'replace').decode('latin-1')
    
    for group in cluster_data:
        # Group Header
        # Color: Expecting tuple (R, G, B)
        r, g, b = group.get('color', (0, 0, 0))
        pdf.set_text_color(r, g, b)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f"Group {group['id']}", 0, 1, 'L')
        
        # Reset text color to black for table
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', 'B', 10)
        
        # Table Header
        # Widths: Model(90), X(40), Y(40) => Total 170
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(90, 8, "Model Name", 1, 0, 'C', 1)
        pdf.cell(50, 8, safe_x_label[:20], 1, 0, 'C', 1) # Truncate if too long
        pdf.cell(50, 8, safe_y_label[:20], 1, 1, 'C', 1)
        
        # Table Rows
        pdf.set_font('Arial', '', 10)
        for item in group['items']:
            # Handle encoding
            model = item['model'].encode('latin-1', 'replace').decode('latin-1')
            x_val = str(item['x']).encode('latin-1', 'replace').decode('latin-1')
            y_val = str(item['y']).encode('latin-1', 'replace').decode('latin-1')
            
            pdf.cell(90, 8, model, 1, 0, 'L')
            pdf.cell(50, 8, x_val, 1, 0, 'C')
            pdf.cell(50, 8, y_val, 1, 1, 'C')
            
        pdf.ln(5)
            
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
