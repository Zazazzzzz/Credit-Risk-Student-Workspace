# ================================================================
# File:        generate_report.py
# Author:      Shazia Ishaq
# Course:      Introduction to Credit Risk — University of Freiburg
# Description: Generates report.pdf directly using ReportLab.
#              Target: exactly 5 pages main body
#              (excluding cover page, table of contents, references)
#
# HOW TO RUN:
#   python Shazia_Ishaq_Final/report/generate_report.py
# ================================================================

import os, numpy as np, pandas as pd
from scipy.stats import norm
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph,
    Spacer, Table, TableStyle, Image, PageBreak, HRFlowable)

# ================================================================
# PATHS
# ================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
OUTPUT_PDF  = os.path.join(SCRIPT_DIR, 'report.pdf')
print("=" * 60)
print("  Generating Credit Portfolio Risk Analysis Report")
print("=" * 60)

# ================================================================
# DATA AND SIMULATION
# ================================================================
params = pd.read_csv(os.path.join(RESULTS_DIR, 'parameters.csv'))
par1 = params[params['portfolio'] == 'Portfolio 1'].dropna(
    subset=['EAD','PD','LGD','rho']).reset_index(drop=True)
par2 = params[params['portfolio'] == 'Portfolio 2'].dropna(
    subset=['EAD','PD','LGD','rho']).reset_index(drop=True)

np.random.seed(42)
N = 100_000

def simulate(table):
    PDs=table['PD'].values; LGDs=table['LGD'].values
    EADs=table['EAD'].values; rhos=table['rho'].values
    m=len(table); w=EADs/EADs.sum(); d=norm.ppf(PDs)
    F=np.random.normal(0,1,N); eps=np.random.normal(0,1,(N,m))
    X=rhos*F[:,None]+np.sqrt(1-rhos**2)*eps
    Y=(X<=d).astype(float); L=Y@(w*LGDs)
    EL=(PDs*LGDs*w).sum(); VaR=np.percentile(L,99)
    ES=L[L>VaR].mean()
    pm=PDs.mean(); vp=0.005*pm*(1-pm); com=pm*(1-pm)/vp-1
    Lbb=np.zeros(N)
    for s in range(N):
        P=np.random.beta(pm*com,(1-pm)*com)
        Yb=np.random.binomial(1,P,m).astype(float)
        Lbb[s]=(Yb*w*LGDs).sum()
    VaRbb=np.percentile(Lbb,99); ESbb=Lbb[Lbb>VaRbb].mean()
    return dict(
        EL_gf=EL, Std_gf=L.std(), VaR_gf=VaR, ES_gf=ES,
        EL_bb=Lbb.mean(), Std_bb=Lbb.std(),
        VaR_bb=VaRbb, ES_bb=ESbb,
        EAD=EADs.sum(), m=m,
        mPD=PDs.mean(), mLGD=LGDs.mean(), mrho=rhos.mean())

r1=simulate(par1); r2=simulate(par2)
re=r2['EL_gf']/r1['EL_gf']; rv=r2['VaR_gf']/r1['VaR_gf']
rs=r2['ES_gf']/r1['ES_gf']
e1=r1['EL_gf']*r1['EAD']/1e6; e2=r2['EL_gf']*r2['EAD']/1e6
ev1=r1['ES_gf']/r1['VaR_gf']; ev2=r2['ES_gf']/r2['VaR_gf']
print(f"  P1 EL={r1['EL_gf']:.6f} VaR={r1['VaR_gf']:.6f} ES={r1['ES_gf']:.6f}")
print(f"  P2 EL={r2['EL_gf']:.6f} VaR={r2['VaR_gf']:.6f} ES={r2['ES_gf']:.6f}")

# ================================================================
# DESIGN CONSTANTS
# ================================================================
DB = colors.HexColor('#1A3A5C')   # dark blue
MB_c = colors.HexColor('#2E6DA4') # mid blue
VL = colors.HexColor('#EBF4FC')   # very light blue
LG = colors.HexColor('#CCCCCC')   # light gray
W  = colors.white
K  = colors.black

PW, PH = A4
ML = MR = 2.54*cm   # 1 inch margins
MT = MB = 2.54*cm
TW = PW - ML - MR   # text width

# ================================================================
# PARAGRAPH STYLES
# ================================================================
def S(n, fn='Times-Roman', sz=12, ld=None, al=TA_JUSTIFY,
      ind=0, sb=0, sa=5, co=K, bold=False):
    return ParagraphStyle(n,
        fontName='Times-Bold' if bold else fn,
        fontSize=sz, leading=ld or round(sz*1.15,1),
        alignment=al, firstLineIndent=ind,
        spaceBefore=sb, spaceAfter=sa, textColor=co)

# Body text — 12pt Times, justified, 1.15 leading, indented
Sb   = S('b',   ind=18, sa=4)
Sni  = S('ni',  ind=0,  sa=4)
# Section headings
Sh2  = S('h2',  sz=13, bold=True, al=TA_LEFT,
          sb=10, sa=4, co=DB)
Sh3  = S('h3',  sz=11, bold=True, al=TA_LEFT,
          sb=6,  sa=2)
# Table caption
Scp  = S('cp',  sz=10, bold=True, al=TA_LEFT, sb=4, sa=2)
# Figure caption — italic centered
Sfc  = S('fc',  sz=9.5, fn='Times-Italic',
          al=TA_CENTER, sb=2, sa=6)
# TOC
Stc  = S('tc',  sz=12, al=TA_LEFT, sa=4)
Stch = S('tch', sz=14, bold=True, sa=10)
# Reference
Srf  = S('rf',  sz=11, al=TA_JUSTIFY, sa=5)
# Cover
Scv  = S('cv',  sz=12, al=TA_CENTER, sa=3)
# Formula
Sfm  = S('fm',  sz=11, fn='Times-Italic',
          al=TA_CENTER, sb=3, sa=3)
# Stats box
Sst  = S('st',  sz=9, fn='Courier', al=TA_LEFT, sa=2)
# TOC page number
Spg  = S('pg',  sz=12, al=TA_RIGHT, sa=4)

# ================================================================
# PAGE HEADER AND FOOTER
# ================================================================
def header_footer(canv, doc):
    """
    Draws on every page after page 2:
    - Header: title left, page number right, thin line below
    - Footer: institution info centered, thin line above
    """
    canv.saveState()
    pg = canv.getPageNumber()
    if pg > 2:
        # ── Header ──────────────────────────────────────────
        y_hdr = PH - MT + 5.5*mm
        canv.setStrokeColor(colors.HexColor('#AAAAAA'))
        canv.setLineWidth(0.5)
        canv.line(ML, y_hdr, PW-MR, y_hdr)
        canv.setFont('Times-Roman', 10)
        canv.setFillColor(colors.HexColor('#333333'))
        canv.drawString(ML, PH-MT+7.5*mm,
                        'Credit Portfolio Risk Analysis')
        canv.drawRightString(PW-MR, PH-MT+7.5*mm, str(pg))
        # ── Footer ──────────────────────────────────────────
        y_ftr = MB - 4*mm
        canv.line(ML, y_ftr, PW-MR, y_ftr)
        canv.setFont('Times-Italic', 9)
        canv.setFillColor(colors.HexColor('#555555'))
        canv.drawCentredString(PW/2, MB-7.5*mm,
            'University of Freiburg  |  '
            'Shazia Ishaq  |  June 2026')
    canv.restoreState()

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def p(text, st=None):
    return Paragraph(text, st or Sb)

def sp(h=4):
    return Spacer(1, h)

def h2(n, t):
    return Paragraph(f'{n}. {t}', Sh2)

def h3(n, t):
    return Paragraph(f'{n} {t}', Sh3)

def embed_figure(fname, width=None, caption=None):
    """Load figure from results folder and embed in report."""
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return [p(f'[Figure not found: {fname}]', Sfc)]
    w = width or TW * 0.88   # figures at 88% of text width
    h = w * 0.48             # aspect ratio
    items = [Image(path, width=w, height=h)]
    if caption:
        items.append(p(caption, Sfc))
    return items

def data_table(headers, rows, col_widths):
    """Create a professionally styled data table."""
    hrow = [Paragraph(h, S('th', sz=9.5, bold=True,
                             co=W, sa=0))
            for h in headers]
    body = [[Paragraph(str(c), S('td', sz=9.5, sa=0))
             for c in row] for row in rows]
    t = Table([hrow]+body, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  DB),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [VL, W]),
        ('GRID',          (0,0), (-1,-1), 0.3, LG),
        ('FONTNAME',      (0,1), (-1,-1), 'Times-Roman'),
        ('FONTSIZE',      (0,1), (-1,-1), 9.5),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 6),
        ('RIGHTPADDING',  (0,0), (-1,-1), 6),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',         (1,1), (-1,-1), 'CENTER'),
    ]))
    return t

def stats_box(text):
    """Highlighted statistics summary box."""
    t = Table([[Paragraph(text, Sst)]], colWidths=[TW])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), VL),
        ('LEFTPADDING',   (0,0),(-1,-1), 8),
        ('RIGHTPADDING',  (0,0),(-1,-1), 8),
        ('TOPPADDING',    (0,0),(-1,-1), 6),
        ('BOTTOMPADDING', (0,0),(-1,-1), 6),
        ('LINEBEFORE',    (0,0),(0,-1),  3, DB),
    ]))
    return t

# ================================================================
# BUILD STORY
# ================================================================
story = []

# ──────────────────────────────────────────────────────────────
# COVER PAGE
# ──────────────────────────────────────────────────────────────
story += [
    sp(5*cm),
    Paragraph('Credit Portfolio Risk Analysis',
              S('ttl', sz=22, bold=True,
                al=TA_CENTER, sa=8)),
    sp(8),
    HRFlowable(width='72%', thickness=1.5, color=K,
               spaceAfter=3, spaceBefore=0),
    HRFlowable(width='72%', thickness=1.5, color=K,
               spaceAfter=22, spaceBefore=0),
    sp(14),
]
for label, value in [
    ('Student',    'Shazia Ishaq'),
    ('Portfolios', 'Portfolio 1 and Portfolio 2'),
    ('Course',
     'Introduction to Credit Risk and Applications in Python'),
    ('Date',       'June 2026'),
]:
    story.append(Paragraph(
        f'<b>{label}:</b>&nbsp;&nbsp;{value}', Scv))
    story.append(sp(2))
story += [
    sp(18),
    HRFlowable(width='72%', thickness=1.5, color=K,
               spaceAfter=3, spaceBefore=0),
    HRFlowable(width='72%', thickness=1.5, color=K,
               spaceAfter=0, spaceBefore=0),
    PageBreak(),
]

# ──────────────────────────────────────────────────────────────
# TABLE OF CONTENTS
# ──────────────────────────────────────────────────────────────
story.append(Paragraph('Contents', Stch))
story.append(sp(4))
for num, title, pg in [
    ('1.', 'Introduction',          '3'),
    ('2.', 'Data',                   '3'),
    ('3.', 'Methodology',            '4'),
    ('4.', 'Results and Discussion', '5'),
    ('5.', 'Conclusion',             '7'),
    ('6.', 'References',             '7'),
]:
    row = Table(
        [[Paragraph(f'{num}&nbsp;&nbsp;{title}', Stc),
          Paragraph(pg, Spg)]],
        colWidths=[TW*0.88, TW*0.12])
    row.setStyle(TableStyle([
        ('LINEBELOW',    (0,0),(0,0), 0.3, LG),
        ('TOPPADDING',   (0,0),(-1,-1), 2),
        ('BOTTOMPADDING',(0,0),(-1,-1), 2),
        ('LEFTPADDING',  (0,0),(-1,-1), 0),
        ('RIGHTPADDING', (0,0),(-1,-1), 0),
    ]))
    story.append(row)
story.append(PageBreak())

# ──────────────────────────────────────────────────────────────
# 1. INTRODUCTION
# ──────────────────────────────────────────────────────────────
story.append(h2(1, 'Introduction'))
story.append(p(
    'Credit portfolio risk quantifies potential losses arising '
    'from the simultaneous default of multiple obligors. Portfolio '
    'credit risk depends on both individual creditworthiness and '
    'the dependence structure among borrowers: when defaults are '
    'positively correlated, adverse macroeconomic conditions can '
    'trigger clustered losses far exceeding individual default '
    'probabilities in isolation (Vasicek, 1987). The Basel III '
    'Internal Ratings-Based (IRB) framework requires banks to '
    'estimate portfolio loss distributions and hold capital '
    'sufficient to absorb unexpected losses at high confidence '
    'levels (Basel Committee, 2006). This report analyzes the '
    'one-year credit risk of two simulated corporate portfolios, '
    'Portfolio 1 (28 Industry A obligors) and Portfolio 2 '
    '(24 Industry B obligors) implementing and comparing '
    'two models. The Gaussian One-Factor Model and the '
    'Beta-Bernoulli mixture model. Risk measures including '
    'Expected Loss (EL), Value at Risk (VaR 99%), and Expected '
    'Shortfall (ES 99%) are computed and interpreted.'))

# ──────────────────────────────────────────────────────────────
# 2. DATA
# ──────────────────────────────────────────────────────────────
story.append(h2(2, 'Data'))
story.append(h3('2.1', 'Dataset Description'))
story.append(p(
    'Six datasets were used in the analysis, covering firm '
    'characteristics, portfolio exposures, market returns, and '
    'historical default information. The <i>entities_info.csv</i> '
    'file provides probability of default (PD), loss given default '
    '(LGD), industry, and region data for each firm, while the '
    'portfolio datasets contain exposure at default (EAD) '
    'information. Daily equity returns from '
    '<i>stock_returns.csv</i> and industry index returns from '
    '<i>industry_index_returns.csv</i> are used to estimate '
    'systematic sensitivity parameters (rho). Historical default '
    'observations from <i>default_history_20y.csv</i> are used to '
    'compare analyst-estimated PDs with observed default experience.'
))

story.append(h3('2.2', 'Cleaning Procedures and Assumptions'))
story.append(p(
    'Entity codes were standardized by removing whitespace and '
    'converting to uppercase to ensure consistent matching across '
    'all files. Industry labels with four inconsistent variants '
    'were mapped to canonical names. Date columns mixing ISO and '
    'European formats were parsed with mixed-format detection. '
    'Missing LGD (6 values) and PD (5 values) were imputed with '
    'sample medians of 0.471 and 2.67% respectively — chosen as '
    'robust estimates resistant to outliers, consistent with '
    'peer group credit quality and standard unsecured corporate '
    'recovery rates. Missing stock returns (2,242 rows) and '
    'rows with missing EAD (2 per portfolio) were dropped, '
    'as financial time-series cannot be reliably imputed '
    'and EAD is a required input for loss calculation. '
    'Missing default events (47 observations) were set to zero, '
    'under the assumption that an absent record indicates no '
    'observed default. Duplicates were removed keeping '
    'the first occurrence across all files.'))

story.append(p(
    'After the cleaning process, the final modelling dataset '
    'contained the characteristics summarized in Table 1.'
))

story.append(sp(2))
story.append(p('Table 1: Dataset Summary after Cleaning', Scp))

story.append(data_table(
    ['Metric', 'Portfolio 1', 'Portfolio 2'],
    [
        ['Obligors',
         str(r1['m']), str(r2['m'])],
        ['Total EAD (EUR M)',
         f"{r1['EAD']/1e6:.1f}", f"{r2['EAD']/1e6:.1f}"],
        ['Mean PD (%)',
         f"{r1['mPD']*100:.3f}", f"{r2['mPD']*100:.3f}"],
        ['PD range (%)',
         f"{par1['PD'].min()*100:.3f} to "
         f"{par1['PD'].max()*100:.3f}",
         f"{par2['PD'].min()*100:.3f} to "
         f"{par2['PD'].max()*100:.3f}"],
        ['Mean LGD (%)',
         f"{r1['mLGD']*100:.1f}", f"{r2['mLGD']*100:.1f}"],
        ['LGD range (%)',
         f"{par1['LGD'].min()*100:.1f} to "
         f"{par1['LGD'].max()*100:.1f}",
         f"{par2['LGD'].min()*100:.1f} to "
         f"{par2['LGD'].max()*100:.1f}"],
        ['Total EL (EUR M)',
         f"{(par1['PD']*par1['LGD']*par1['EAD']).sum()/1e6:.4f}",
         f"{(par2['PD']*par2['LGD']*par2['EAD']).sum()/1e6:.4f}"],
        ['Mean rho',
         f"{r1['mrho']:.3f}", f"{r2['mrho']:.3f}"],
        ['rho range',
         f"{par1['rho'].min():.3f} to "
         f"{par1['rho'].max():.3f}",
         f"{par2['rho'].min():.3f} to "
         f"{par2['rho'].max():.3f}"],
        ['Industry',    'Industry A', 'Industry B'],
        ['Regions',
         'Asia, Europe, N./S. America',
         'Asia, Europe, N./S. America'],
    ],
    [TW*0.44, TW*0.28, TW*0.28]))

# ──────────────────────────────────────────────────────────────
# 3. METHODOLOGY
# ──────────────────────────────────────────────────────────────
story.append(h2(3, 'Methodology'))
story.append(h3('3.1', 'Literature Review'))
story.append(p(
    'Vasicek (1987) derived the Gaussian One-Factor Model, '
    'providing a closed-form portfolio loss distribution under the '
    'assumption that firm asset returns follow a single-factor '
    'Gaussian model. The Basel Committee (2006) adopted this '
    'framework as the theoretical basis of the IRB approach, '
    'where banks estimate firm-level PD and LGD while a '
    'fixed asset correlation governs default dependence. '
    'Fitch Ratings (2008) examined the empirical validity of '
    'Basel II correlation assumptions by fitting Beta distributions '
    'to historical loss-rate data and inverting the IRB formula '
    'to obtain implied correlations, finding that regulatory '
    'values are conservative relative to historical experience. '
    'Merton (1974) establishes the theoretical foundation for '
    'using equity return correlations as proxies for asset '
    'correlations, since equity prices reflect firm asset value '
    'dynamics under the structural credit risk framework.'))

story.append(h3('3.2', 'Gaussian One-Factor Model'))

story.append(p(
    'The Gaussian One-Factor Model is a standard framework for '
    'portfolio credit risk measurement and forms the basis of '
    'the Basel Internal Ratings-Based (IRB) approach. The model '
    'captures default dependence through a common systematic '
    'factor that affects all obligors, while firm-specific shocks '
    'represent idiosyncratic risk.'))

story.append(p(
    '<i>X<sub>i</sub> = &#961;<sub>i</sub> &#183; F + '
    '&#8730;(1&#8722;&#961;<sub>i</sub><sup>2</sup>)'
    '&#183;&#949;<sub>i</sub>,&#8194;'
    'F,&#949;<sub>i</sub> &#126; N(0,1)</i>',
    Sfm))

story.append(p(
    'where <i>F</i> is the common economic factor and '
    '<i>&#961;<sub>i</sub></i> represents the systematic '
    'sensitivity of firm <i>i</i>. A default occurs when the '
    'latent variable falls below the threshold implied by the '
    'firm\'s probability of default (PD). Portfolio losses are '
    'calculated using simulated default outcomes together with '
    'exposure at default (EAD) and loss given default (LGD). '
    'The model was selected because it is widely used in practice '
    'and provides a transparent framework for measuring portfolio '
    'credit risk.'))

story.append(h3('3.3', 'Beta-Bernoulli Model'))

story.append(p(
    'The Beta-Bernoulli model treats the portfolio default '
    'probability as a random variable, allowing uncertainty in '
    'default rates to be incorporated directly into the loss '
    'simulation. Dependence between obligors arises through a '
    'shared portfolio-wide default probability rather than a '
    'common economic factor.'))

story.append(p(
    '<i>P &#126; Beta(&#945;, &#946;),&#8194;'
    'Y<sub>i</sub> | P &#126; Bernoulli(P)</i>',
    Sfm))

story.append(p(
    'The parameters &#945; and &#946; are estimated using the '
    'portfolio mean PD and an assumed default correlation of '
    '0.5% through the method of moments. The model was selected '
    'as an alternative dependence framework to compare with the '
    'Gaussian One-Factor Model and to evaluate the sensitivity '
    'of risk measures to different modelling assumptions.'))

story.append(h3('3.4', 'Parameter Estimation'))
story.append(p(
    '<b>Probability of Default.</b> '
    'PD is taken from base_pd_hint representing analyst-estimated '
    ' one-year default probabilities. '
    'Validation against 20-year empirical default rates yields '
    'a Pearson correlation of 0.366 (Figure 2), confirming '
    'broad consistency with historical experience. '
    'The empirical PD is used for validation only, as many '
    'firms have zero historical defaults over 20 years, a '
    'known limitation of short records for low-default '
    'portfolios.'))
story.append(p(
    '<b>Systematic Sensitivity (rho).</b> '
    'The parameter &#961;<sub>i</sub> is estimated as the '
    'Pearson correlation between each firm\'s daily equity '
    'return and its industry index return, following Merton '
    '(1974). Estimates are clipped to [0.1, 0.9] to ensure '
    'valid model inputs. The Basel Committee (2006) fixes '
    '&#961; between 0.12 and 0.24 for regulatory IRB capital; '
    'our data-driven estimates are higher (mean 0.441 and 0.540 '
    'for Portfolios 1 and 2), consistent with Fitch Ratings '
    '(2008) who find that equity-return correlations tend to '
    'exceed asset correlations implied by historical loss data, '
    'reflecting leverage effects. '
    '<b>Simulation</b> uses 100,000 Monte Carlo scenarios '
    '(seed = 42) at 99% confidence, consistent with Basel III.'))

# Figures 1 and 2 — side by side to save space
fig1_path = os.path.join(RESULTS_DIR, 'rho_distribution.png')
fig2_path = os.path.join(RESULTS_DIR, 'pd_validation.png')
if os.path.exists(fig1_path) and os.path.exists(fig2_path):
    fw = TW*0.495
    fh = fw*0.62
    fig_row = Table(
        [[Image(fig1_path, width=fw, height=fh),
          Image(fig2_path, width=fw, height=fh)]],
        colWidths=[TW*0.50, TW*0.50])
    fig_row.setStyle(TableStyle([
        ('TOPPADDING',   (0,0),(-1,-1), 4),
        ('BOTTOMPADDING',(0,0),(-1,-1), 2),
        ('LEFTPADDING',  (0,0),(-1,-1), 0),
        ('RIGHTPADDING', (0,0),(-1,-1), 0),
        ('VALIGN',       (0,0),(-1,-1), 'TOP'),
    ]))
    story.append(sp(4))
    story.append(fig_row)
    cap_row = Table(
        [[Paragraph(
            'Figure 1: Distribution of estimated rho by portfolio. '
            'Pearson correlation of firm equity return with '
            'industry index return.', Sfc),
          Paragraph(
            'Figure 2: PD validation — base_pd_hint versus '
            'empirical PD from 20-year default history '
            '(correlation = 0.366).', Sfc)]],
        colWidths=[TW*0.50, TW*0.50])
    cap_row.setStyle(TableStyle([
        ('TOPPADDING',   (0,0),(-1,-1), 0),
        ('BOTTOMPADDING',(0,0),(-1,-1), 0),
        ('LEFTPADDING',  (0,0),(-1,-1), 0),
        ('RIGHTPADDING', (0,0),(-1,-1), 0),
    ]))
    story.append(cap_row)
else:
    for item in embed_figure('rho_distribution.png',
        caption='Figure 1: Estimated rho distribution.'):
        story.append(item)
    for item in embed_figure('pd_validation.png',
        caption='Figure 2: PD validation.'):
        story.append(item)

# ──────────────────────────────────────────────────────────────
# 4. RESULTS AND DISCUSSION
# ──────────────────────────────────────────────────────────────
story.append(h2(4, 'Results and Discussion'))
story.append(h3('4.1', 'Risk Measures'))
story.append(sp(2))
story.append(p('Table 2: Risk Measures by Portfolio and Model '
                '(100,000 simulations | 99% confidence | seed = 42)',
                Scp))
story.append(data_table(
    ['Measure', 'P1 Gaussian', 'P1 Beta-Bernoulli',
     'P2 Gaussian', 'P2 Beta-Bernoulli'],
    [['EL',
      f"{r1['EL_gf']:.6f}", f"{r1['EL_bb']:.6f}",
      f"{r2['EL_gf']:.6f}", f"{r2['EL_bb']:.6f}"],
     ['Std',
      f"{r1['Std_gf']:.6f}", f"{r1['Std_bb']:.6f}",
      f"{r2['Std_gf']:.6f}", f"{r2['Std_bb']:.6f}"],
     ['VaR 99%',
      f"{r1['VaR_gf']:.6f}", f"{r1['VaR_bb']:.6f}",
      f"{r2['VaR_gf']:.6f}", f"{r2['VaR_bb']:.6f}"],
     ['ES 99%',
      f"{r1['ES_gf']:.6f}", f"{r1['ES_bb']:.6f}",
      f"{r2['ES_gf']:.6f}", f"{r2['ES_bb']:.6f}"]],
    [TW*0.17, TW*0.2075, TW*0.2075,
     TW*0.2075, TW*0.2075]))
story.append(sp(3))
story.append(stats_box(
    f'Gaussian ratios (P2 / P1):  '
    f'EL = {re:.2f}  |  VaR = {rv:.2f}  |  ES = {rs:.2f}\n'
    f'Absolute EL:  '
    f'P1 = EUR {e1:.2f}M ({r1["EL_gf"]*100:.3f}% of EAD)  |  '
    f'P2 = EUR {e2:.2f}M ({r2["EL_gf"]*100:.3f}% of EAD)'))
story.append(sp(4))

story.append(h3('4.2', 'Portfolio Comparison'))
story.append(p(
    f'Portfolio 2 (Industry B) exhibits substantially higher '
    f'credit risk across all measures and under both models. '
    f'Under the Gaussian One-Factor Model, Portfolio 2 yields '
    f'an Expected Loss of {r2["EL_gf"]*100:.2f}% of total EAD '
    f'(EUR {e2:.2f}M) compared with {r1["EL_gf"]*100:.2f}% '
    f'(EUR {e1:.2f}M) for Portfolio 1, a difference of '
    f'{re:.2f} times. This is driven directly by the higher '
    f'mean PD of Industry B firms (3.311% versus 2.034%) and '
    f'their higher mean LGD (49.5% versus 46.8%), since the '
    f'analytical formula EL = Σ PDi · LGDi · wi' 'is a linear function of both '
    f'parameters. The tail risk difference is more pronounced: '
    f'VaR 99% for Portfolio 2 ({r2["VaR_gf"]*100:.2f}% of EAD) '
    f'exceeds Portfolio 1 ({r1["VaR_gf"]*100:.2f}%) by '
    f'{rv:.2f} times, and ES 99% ({r2["ES_gf"]*100:.2f}% '
    f'versus {r1["ES_gf"]*100:.2f}%) by {rs:.2f} times. '
    f'The gap between EL and VaR of approximately 8 to 9 times '
    f'reflects the right-skewed nature of credit loss '
    f'distributions: most scenarios produce near-zero losses '
    f'while rare adverse conditions generate '
    f'disproportionately large losses. '
    f'Portfolio 2 firms exhibit higher systematic sensitivity '
    f'(mean rho = 0.540 versus 0.441), which strengthens the '
    f'link to the common economic factor and widens the tail '
    f'without affecting EL, since the EL formula is independent '
    f'of rho (Vasicek, 1987). ES to VaR ratios of {ev1:.2f} '
    f'and {ev2:.2f} for Portfolios 1 and 2 respectively '
    f'confirm heavy-tailed distributions in both cases.'))

# Figures 3 and 4 — side by side
fig3_path = os.path.join(RESULTS_DIR, 'loss_distributions.png')
fig4_path = os.path.join(RESULTS_DIR, 'comparison_chart.png')
if os.path.exists(fig3_path) and os.path.exists(fig4_path):
    fw = TW*0.495; fh = fw*0.62
    fr = Table(
        [[Image(fig3_path, width=fw, height=fh),
          Image(fig4_path, width=fw, height=fh)]],
        colWidths=[TW*0.50, TW*0.50])
    fr.setStyle(TableStyle([
        ('TOPPADDING',   (0,0),(-1,-1), 4),
        ('BOTTOMPADDING',(0,0),(-1,-1), 2),
        ('LEFTPADDING',  (0,0),(-1,-1), 0),
        ('RIGHTPADDING', (0,0),(-1,-1), 0),
        ('VALIGN',       (0,0),(-1,-1), 'TOP'),
    ]))
    story.append(sp(4))
    story.append(fr)
    cr = Table(
        [[Paragraph(
            'Figure 3: Simulated loss distributions — '
            'Gaussian (top) vs Beta-Bernoulli (bottom). '
            'Green = EL | Orange = VaR 99% | Red = ES 99%.', Sfc),
          Paragraph(
            'Figure 4: Risk measure comparison across '
            'both portfolios and both models.', Sfc)]],
        colWidths=[TW*0.50, TW*0.50])
    cr.setStyle(TableStyle([
        ('TOPPADDING',   (0,0),(-1,-1), 0),
        ('BOTTOMPADDING',(0,0),(-1,-1), 0),
        ('LEFTPADDING',  (0,0),(-1,-1), 0),
        ('RIGHTPADDING', (0,0),(-1,-1), 0),
    ]))
    story.append(cr)
else:
    for item in embed_figure('loss_distributions.png',
        caption='Figure 3: Simulated loss distributions.'):
        story.append(item)
    for item in embed_figure('comparison_chart.png',
        caption='Figure 4: Risk measure comparison.'):
        story.append(item)

story.append(h3('4.3', 'Model Comparison'))
story.append(p(
    f'Both models yield closely aligned EL estimates for each '
    f'portfolio — Portfolio 1: {r1["EL_gf"]*100:.4f}% (Gaussian) '
    f'versus {r1["EL_bb"]*100:.4f}% (Beta-Bernoulli); Portfolio 2: '
    f'{r2["EL_gf"]*100:.4f}% versus {r2["EL_bb"]*100:.4f}%  '
    f'confirming that EL is invariant to the dependence structure '
    f'and depends solely on PD, LGD, and EAD. The models diverge '
    f'substantially in tail risk: Gaussian VaR 99% exceeds '
    f'Beta-Bernoulli VaR 99% by 40% for Portfolio 1 and 82% for '
    f'Portfolio 2, with ES differences of 51% and 97% '
    f'respectively. The Gaussian model generates higher tail risk '
    f'because the estimated rho values (mean 0.441 and 0.540) '
    f'translate into strong systematic dependence through the '
    f'common factor, while the Beta-Bernoulli model parameterizes '
    f'dependence through a default correlation of only 0.5%. '
    f'This divergence illustrates model risk in credit portfolio '
    f'measurement: the choice of dependence structure can alter '
    f'capital requirement estimates by a factor approaching two. '
    f'Consistent with Fitch Ratings (2008), our equity-return '
    f'based correlations exceed values implied by historical loss '
    f'data, reflecting leverage effects and the conservative '
    f'calibration of the Basel regulatory framework.'))

# ──────────────────────────────────────────────────────────────
# 5. CONCLUSION
# ──────────────────────────────────────────────────────────────
story.append(h2(5, 'Conclusion'))
story.append(p(
    f'This report quantified the one-year credit risk of two '
    f'simulated corporate portfolios using the Gaussian '
    f'One-Factor Model and the Beta-Bernoulli model applied to '
    f'systematically cleaned and merged datasets. Three principal '
    f'findings emerge. First, Portfolio 2 (Industry B) exhibits '
    f'materially higher risk than Portfolio 1 (Industry A) across '
    f'all measures, EL {re:.2f} times higher, VaR 99% {rv:.2f} '
    f'times higher, and ES 99% {rs:.2f} times higher driven by '
    f'elevated PD, LGD, and systematic sensitivity parameters. '
    f'Second, both models produce consistent EL estimates but '
    f'diverge by 40 to 97% in tail risk measures, illustrating '
    f'that capital requirements are sensitive to the choice of '
    f'dependence model. Third, ES to VaR ratios of {ev2:.2f} '
    f'to {ev1:.2f} confirm heavy-tailed loss distributions, '
    f'underscoring ES as the preferred regulatory risk measure '
    f'over VaR. Limitations include the short rho estimation '
    f'window of approximately two years, which may understate '
    f'stress-period correlations (Fitch Ratings, 2008), and '
    f'reliance on analyst PD estimates that may not fully reflect '
    f'current market conditions. Future work could incorporate '
    f'stress-tested correlation scenarios and time-varying PD '
    f'models calibrated to credit spreads.'))

# ──────────────────────────────────────────────────────────────
# 6. REFERENCES
# ──────────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(h2(6, 'References'))
story.append(sp(6))
for ref in [
    ('Basel Committee on Banking Supervision. (2006). '
     '<i>International Convergence of Capital Measurement '
     'and Capital Standards: A Revised Framework</i>. '
     'Bank for International Settlements. '
     'https://www.bis.org/publ/bcbs128.htm'),
    ('Fitch Ratings. (2008). <i>Understanding Fitch\'s '
     'Empirical Approach to Basel II Correlation Values</i>. '
     'Fitch Ratings Special Report.'),
    ('Merton, R. C. (1974). On the pricing of corporate debt: '
     'The risk structure of interest rates. '
     '<i>Journal of Finance</i>, <i>29</i>(2), 449&#8211;470. '
     'https://doi.org/10.1111/j.1540-6261.1974.tb03058.x'),
    ('Vasicek, O. (1987). <i>Probability of Loss on Loan '
     'Portfolio</i>. KMV Corporation Working Paper.'),
]:
    story.append(p(ref, Srf))
    story.append(sp(4))

# ================================================================
# BUILD PDF
# ================================================================
print("\nBuilding PDF...")
doc = BaseDocTemplate(
    OUTPUT_PDF, pagesize=A4,
    leftMargin=ML, rightMargin=MR,
    topMargin=MT + 12*mm, bottomMargin=MB + 10*mm,
    title='Credit Portfolio Risk Analysis',
    author='Shazia Ishaq',
    subject='Credit Risk Final Assignment — University of Freiburg')
frame = Frame(ML, MB+10*mm, TW, PH-MT-MB-22*mm, id='main')
doc.addPageTemplates([
    PageTemplate(id='main', frames=[frame],
                 onPage=header_footer)])
doc.build(story)
print(f"\nReport saved: {os.path.normpath(OUTPUT_PDF)}")
print("Done!")
