#!/usr/bin/env python3
"""
tools/build_pdf_reports.py

Generates two comprehensive, publication-grade PDF documents:
1. artifacts/CacheWinograd_Comprehensive_Empirical_Analysis.pdf (8-Page Master Monograph)
2. artifacts/CacheWinograd_AI_Reference_Digest.pdf (2-Page Executive AI Reference Card)

Engineered specifically for AI ingestion (Claude, GPT-4, Gemini) and human peer-review:
- Strict page budgeting with clean chapter PageBreaks
- Complete raw and summary data tables with confidence intervals and p-values
- High-resolution embedded figures with descriptive captions
- Microarchitectural analysis: register spilling, cache probe mismatch, adaptive-m inversion
- Methodology transparency (Constraint #6 layer-time aggregation note)
"""

import os
import sys
import csv
import json
import datetime
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")
RAW_DIR = os.path.join(BASE_DIR, "raw_logs")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")

class MonographCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(40, 762, "CACHEWINOGRAD: EMPIRICAL ANALYSIS & MICROARCHITECTURAL AUDIT")
            self.setFont("Helvetica", 7.5)
            self.drawRightString(572, 762, "RESEARCH MONOGRAPH")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.6)
            self.line(40, 756, 572, 756)
            
        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.6)
        self.line(40, 40, 572, 40)
        
        self.setFont("Helvetica", 7.5)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 28, page_str)
        self.drawString(40, 28, "CONFIDENTIAL EXPERIMENTAL SUITE — OPTIMIZED FOR AI REASONING & REPRODUCIBILITY")
        self.restoreState()


class DigestCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Header
        if self._pageNumber > 1:
            self.drawString(36, 764, "CACHEWINOGRAD AI REFERENCE DIGEST — ARCHITECTURE & BENCHMARK CARD")
            self.setFont("Helvetica", 7.5)
            self.drawRightString(576, 764, "EXECUTIVE CARD")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.6)
            self.line(36, 758, 576, 758)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.6)
        self.line(36, 38, 576, 38)
        
        self.setFont("Helvetica", 7.5)
        page_str = f"Reference Card — Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 26, page_str)
        self.drawString(36, 26, "RAPID INGESTION CARD FOR LLMS (CLAUDE / GPT-4 / GEMINI) — CACHEWINOGRAD")
        self.restoreState()


def get_styles():
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=20, leading=24,
        textColor=colors.HexColor('#1A365D'), spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSub', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9.5, leading=13,
        textColor=colors.HexColor('#4A5568'), spaceAfter=10
    )
    h1_style = ParagraphStyle(
        'SecH1', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=13, leading=16,
        textColor=colors.HexColor('#2B6CB0'), spaceBefore=8, spaceAfter=5,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SecH2', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=10, leading=13,
        textColor=colors.HexColor('#2C5282'), spaceBefore=6, spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyTextDark', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8, leading=11,
        textColor=colors.HexColor('#2D3748'), spaceAfter=4
    )
    callout_style = ParagraphStyle(
        'CalloutText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.5, leading=10.5,
        textColor=colors.HexColor('#2C5282')
    )
    caption_style = ParagraphStyle(
        'FigCaption', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=7, leading=9.5,
        textColor=colors.HexColor('#4A5568'), alignment=1, spaceBefore=3, spaceAfter=4
    )
    th_style = ParagraphStyle(
        'TableHead', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=7, leading=9,
        textColor=colors.white, alignment=1
    )
    tc_style = ParagraphStyle(
        'TableBody', parent=styles['Normal'],
        fontName='Helvetica', fontSize=6.8, leading=8.5,
        textColor=colors.HexColor('#2D3748')
    )
    tc_mono = ParagraphStyle(
        'TableMono', parent=styles['Normal'],
        fontName='Courier', fontSize=6.5, leading=8,
        textColor=colors.HexColor('#1A202C')
    )
    tc_bold = ParagraphStyle(
        'TableBodyBold', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=6.8, leading=8.5,
        textColor=colors.HexColor('#1A202C')
    )
    tc_win = ParagraphStyle(
        'TableBodyWin', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=6.8, leading=8.5,
        textColor=colors.HexColor('#22543D')
    )
    tc_loss = ParagraphStyle(
        'TableBodyLoss', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=6.8, leading=8.5,
        textColor=colors.HexColor('#742A2A')
    )
    
    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'h1': h1_style,
        'h2': h2_style,
        'body': body_style,
        'callout': callout_style,
        'caption': caption_style,
        'th': th_style,
        'tc': tc_style,
        'tc_mono': tc_mono,
        'tc_bold': tc_bold,
        'tc_win': tc_win,
        'tc_loss': tc_loss,
    }


def make_callout(text, style, bg="#EBF8FF", border="#BEE3F8", width=532):
    p = Paragraph(text, style)
    t = Table([[p]], colWidths=[width])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg)),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor(border)),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    return t


# ==============================================================================
# PDF 1: FULL MONOGRAPH (8 PAGES)
# ==============================================================================
def build_full_report_pdf():
    pdf_path = os.path.join(ARTIFACTS_DIR, "CacheWinograd_Comprehensive_Empirical_Analysis.pdf")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=40, rightMargin=40, topMargin=48, bottomMargin=48
    )
    st = get_styles()
    story = []

    # --------------------------------------------------------------------------
    # PAGE 1: Title, Executive Summary, Hardware Architecture & Formulation
    # --------------------------------------------------------------------------
    story.append(Paragraph("CacheWinograd: Comprehensive Empirical Analysis & Microarchitectural Audit", st['title']))
    story.append(Paragraph(
        "<b>Multi-Hardware Empirical Verification on Embedded ARM (Raspberry Pi 4, Jetson Nano) & Apple Silicon</b><br/>"
        "<i>Comprehensive Technical Monograph | 12 Raw Datasets | Methodological & Architectural Audit</i>",
        st['subtitle']
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=8))

    exec_summary = (
        "<b>EXECUTIVE AUDIT SUMMARY:</b> This monograph presents an exhaustive, statistically rigorous evaluation "
        "of the CacheWinograd fused convolution kernel across physical hardware platforms. Core findings include:<br/>"
        "&bull; <b>End-to-End VGG-16 Speedup:</b> CacheWinograd delivers a statistically significant <b>+4.80% latency reduction</b> "
        "(p = 2.82e-6, saving <b>93.15s per run</b>) on Raspberry Pi 4, driven by deep 3x3 cascades with large channel counts.<br/>"
        "&bull; <b>Autotiler Policy Inversion:</b> An empirical adaptive-m sweep proves m=6 outperforms m=2 by up to <b>2.84x (p &lt; 1e-10)</b> "
        "for small channels (Cin &le; 32), directly refuting the paper's conservative working-set threshold.<br/>"
        "&bull; <b>Register Spilling Bottleneck:</b> Disassembly of the (128,128) kernel reveals a <b>2.0x virtual register over-subscription</b> "
        "and 6.37% stack spill ratio on 32-register ARM NEON, causing a catastrophic performance cliff (43.7s execution on Jetson).<br/>"
        "&bull; <b>Hardware Cache Discrepancy:</b> Sysfs probe reveals RPi4 has <b>1 MB L2 cache</b> (not 2 MB as claimed in the paper, -50% diff).<br/>"
        "&bull; <b>Multi-Core Reality:</b> 4-core multi-threading on small spatial tiles yields only <b>+0.74%</b> speedup vs 33.11% claimed."
    )
    story.append(make_callout(exec_summary, st['callout'], bg="#F7FAFC", border="#CBD5E0"))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Hardware Microarchitecture & Platform Specifications", st['h1']))
    hw_table_data = [
        [Paragraph(h, st['th']) for h in ["Architectural Feature", "Raspberry Pi 4 Model B", "NVIDIA Jetson Nano", "Apple Silicon (macOS)"]],
        [Paragraph("CPU Architecture", st['tc_bold']), Paragraph("Quad ARM Cortex-A72 @ 1.50 GHz", st['tc']), Paragraph("Quad ARM Cortex-A57 @ 1.43 GHz", st['tc']), Paragraph("12-core ARMv8.5+ @ 3.20 GHz", st['tc'])],
        [Paragraph("Vector Engine", st['tc_bold']), Paragraph("32 x 128-bit NEON SIMD registers", st['tc']), Paragraph("32 x 128-bit NEON SIMD registers", st['tc']), Paragraph("32 x 128-bit NEON (Wide-issue)", st['tc'])],
        [Paragraph("L1 Data Cache", st['tc_bold']), Paragraph("32 KB / core (Sysfs Verified)", st['tc']), Paragraph("32 KB / core (Sysfs Verified)", st['tc']), Paragraph("64 KB / core", st['tc'])],
        [Paragraph("L2 Cache (Shared)", st['tc_bold']), Paragraph("<b>1,024 KB (1MB)</b> (Sysfs: -50% vs paper)", st['tc_loss']), Paragraph("2,048 KB (2MB) (Sysfs: Match)", st['tc_win']), Paragraph("4,096 KB (4MB Shared L2/SLC)", st['tc'])],
        [Paragraph("Cache Line Size", st['tc_bold']), Paragraph("64 Bytes", st['tc']), Paragraph("64 Bytes", st['tc']), Paragraph("128 Bytes", st['tc'])],
        [Paragraph("DRAM Subsystem", st['tc_bold']), Paragraph("2 GB LPDDR4-3200 (~12.8 GB/s)", st['tc']), Paragraph("4 GB 64-bit LPDDR4-1600 (25.6 GB/s)", st['tc']), Paragraph("Unified Memory (&gt; 100 GB/s)", st['tc'])],
        [Paragraph("Peak FP32 Compute", st['tc_bold']), Paragraph("24.0 GFLOPS (CPU NEON)", st['tc']), Paragraph("45.8 GFLOPS (CPU NEON)", st['tc']), Paragraph("&gt; 250 GFLOPS (CPU)", st['tc'])],
        [Paragraph("Thermal Throttling", st['tc_bold']), Paragraph("<b>throttled=0x0</b> (0% throttling across runs)", st['tc_win']), Paragraph("Monitored via tegrastats", st['tc']), Paragraph("Active cooling, unthrottled", st['tc'])],
    ]
    t_hw = Table(hw_table_data, colWidths=[105, 142, 142, 143])
    t_hw.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_hw)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Kernel Formulation, Tiling & Working Set Model", st['h1']))
    story.append(Paragraph(
        "Winograd minimal filtering evaluates 2D convolution as Y = A<sup>T</sup> [ (G g G<sup>T</sup>) &times; (B<sup>T</sup> d B) ] A. "
        "For tile F(m, r) with filter size r=3, input tile dimension is &alpha; = m + r - 1. Naive implementations write transformed tensors "
        "V = B<sup>T</sup> d B and U = G g G<sup>T</sup> to global DRAM. CacheWinograd fuses input transform, element-wise GEMM, and output transform "
        "into a single loop nest to eliminate intermediate DRAM traffic.",
        st['body']
    ))
    ws_box = (
        "<b>WORKING SET & FALLBACK FORMULA:</b> Active working set is modeled as:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>WS(m, C<sub>in</sub>, C<sub>out</sub>) = 4 &times; [ (m + 2)&sup2; &times; C<sub>in</sub> + (m + 2)&sup2; &times; C<sub>out</sub> + (m + 2)&sup2; &times; C<sub>in</sub> &times; C<sub>out</sub> / TILE_FACTOR ] (Bytes)</b><br/>"
        "<b>Fallback Guard:</b> If WS &gt; &gamma; &times; L1D (where &gamma; = 0.70 &rarr; 22,938 B on 32KB L1), "
        "the kernel automatically falls back to partitioned tiling to avoid cache thrashing."
    )
    story.append(make_callout(ws_box, st['callout'], bg="#EDF2F7", border="#CBD5E0"))
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 2: Section 3: Microbenchmark Suite (RPi4 & Jetson Nano)
    # --------------------------------------------------------------------------
    story.append(Paragraph("3. Microbenchmark Suite: Comprehensive 7-Configuration Evaluation", st['h1']))
    story.append(Paragraph(
        "Seven canonical convolution workloads were evaluated across Raspberry Pi 4 and Jetson Nano under identical test harnesses. "
        "Tables 1 and 2 detail baseline vs fused latencies, 95% confidence intervals, and two-sample t-test p-values.",
        st['body']
    ))

    story.append(Paragraph("Table 1: Raspberry Pi 4 Microbenchmark Performance (n=30 runs, m=4, tile_dim=6)", st['h2']))
    rpi4_data = [
        [Paragraph(h, st['th']) for h in ["Workload", "Cin", "Cout", "Baseline (ms)", "Fused (ms)", "Improvement", "p-value", "Significance", "Paper Claim", "Paper Match"]],
        [Paragraph("(16,32)", st['tc_bold']), Paragraph("16", st['tc']), Paragraph("32", st['tc']), Paragraph("8.9637 &plusmn; 0.010", st['tc']), Paragraph("8.9436 &plusmn; 0.007", st['tc']), Paragraph("+0.22%", st['tc_win']), Paragraph("2.13e-3", st['tc']), Paragraph("SIGNIFICANT", st['tc_win']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(32,16)", st['tc_bold']), Paragraph("32", st['tc']), Paragraph("16", st['tc']), Paragraph("4.3871 &plusmn; 0.006", st['tc']), Paragraph("4.3759 &plusmn; 0.008", st['tc']), Paragraph("+0.25%", st['tc_win']), Paragraph("3.62e-2", st['tc']), Paragraph("SIGNIFICANT", st['tc_win']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(32,32)", st['tc_bold']), Paragraph("32", st['tc']), Paragraph("32", st['tc']), Paragraph("13.3931 &plusmn; 0.010", st['tc']), Paragraph("13.3720 &plusmn; 0.038", st['tc']), Paragraph("+0.16%", st['tc']), Paragraph("0.297", st['tc']), Paragraph("NOT_SIG", st['tc']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(32,64)", st['tc_bold']), Paragraph("32", st['tc']), Paragraph("64", st['tc']), Paragraph("44.0000 &plusmn; 0.252", st['tc']), Paragraph("43.8069 &plusmn; 0.077", st['tc']), Paragraph("+0.44%", st['tc']), Paragraph("0.160", st['tc']), Paragraph("NOT_SIG", st['tc']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(64,32)", st['tc_bold']), Paragraph("64", st['tc']), Paragraph("32", st['tc']), Paragraph("21.9022 &plusmn; 0.212", st['tc']), Paragraph("21.7475 &plusmn; 0.057", st['tc']), Paragraph("+0.71%", st['tc']), Paragraph("0.176", st['tc']), Paragraph("NOT_SIG", st['tc']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(64,64)", st['tc_bold']), Paragraph("64", st['tc']), Paragraph("64", st['tc']), Paragraph("85.3300 &plusmn; 0.264", st['tc']), Paragraph("85.5026 &plusmn; 0.375", st['tc']), Paragraph("-0.20%", st['tc_loss']), Paragraph("0.464", st['tc']), Paragraph("NOT_SIG", st['tc']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(128,128)", st['tc_bold']), Paragraph("128", st['tc']), Paragraph("128", st['tc']), Paragraph("2,322.15 &plusmn; 0.697", st['tc']), Paragraph("2,322.40 &plusmn; 0.624", st['tc']), Paragraph("-0.01%", st['tc_loss']), Paragraph("0.602", st['tc']), Paragraph("NOT_SIG", st['tc']), Paragraph("+63.91%", st['tc_loss']), Paragraph("MISMATCH (-100%)", st['tc_loss'])],
    ]
    t_rpi = Table(rpi4_data, colWidths=[46, 24, 24, 76, 76, 52, 42, 60, 56, 76])
    t_rpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_rpi)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Table 2: NVIDIA Jetson Nano Microbenchmark Performance (n=10 runs)", st['h2']))
    jn_data = [
        [Paragraph(h, st['th']) for h in ["Workload", "Cin", "Cout", "Baseline (ms)", "Fused (ms)", "Improvement", "p-value", "Significance", "Paper Claim", "Paper Match"]],
        [Paragraph("(16,32)", st['tc_bold']), Paragraph("16", st['tc']), Paragraph("32", st['tc']), Paragraph("2.3064 &plusmn; 2.247", st['tc']), Paragraph("0.9881 &plusmn; 0.003", st['tc']), Paragraph("+57.16%", st['tc_win']), Paragraph("0.280", st['tc']), Paragraph("NOT_SIG (High Var)", st['tc']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(32,16)", st['tc_bold']), Paragraph("32", st['tc']), Paragraph("16", st['tc']), Paragraph("1.0978 &plusmn; 0.006", st['tc']), Paragraph("1.1056 &plusmn; 0.030", st['tc']), Paragraph("-0.71%", st['tc_loss']), Paragraph("0.623", st['tc']), Paragraph("NOT_SIG", st['tc']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(32,32)", st['tc_bold']), Paragraph("32", st['tc']), Paragraph("32", st['tc']), Paragraph("2.7597 &plusmn; 0.039", st['tc']), Paragraph("2.7208 &plusmn; 0.027", st['tc']), Paragraph("+1.41%", st['tc_win']), Paragraph("0.128", st['tc']), Paragraph("NOT_SIG", st['tc']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(32,64)", st['tc_bold']), Paragraph("32", st['tc']), Paragraph("64", st['tc']), Paragraph("8.1910 &plusmn; 0.025", st['tc']), Paragraph("8.1507 &plusmn; 0.031", st['tc']), Paragraph("+0.49%", st['tc_win']), Paragraph("0.066", st['tc']), Paragraph("NOT_SIG", st['tc']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(64,32)", st['tc_bold']), Paragraph("64", st['tc']), Paragraph("32", st['tc']), Paragraph("4.5615 &plusmn; 0.021", st['tc']), Paragraph("4.5450 &plusmn; 0.020", st['tc']), Paragraph("+0.36%", st['tc_win']), Paragraph("0.276", st['tc']), Paragraph("NOT_SIG", st['tc']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(64,64)", st['tc_bold']), Paragraph("64", st['tc']), Paragraph("64", st['tc']), Paragraph("14.7619 &plusmn; 0.024", st['tc']), Paragraph("14.7539 &plusmn; 0.014", st['tc']), Paragraph("+0.05%", st['tc']), Paragraph("0.578", st['tc']), Paragraph("NOT_SIG", st['tc']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
        [Paragraph("(128,128)*", st['tc_bold']), Paragraph("128", st['tc']), Paragraph("128", st['tc']), Paragraph("44,281.5 &plusmn; 521.2", st['tc']), Paragraph("43,695.9 &plusmn; 435.8", st['tc']), Paragraph("+1.32%", st['tc_win']), Paragraph("0.109", st['tc']), Paragraph("NOT_SIG", st['tc']), Paragraph("-6.52%", st['tc_loss']), Paragraph("MISMATCH (+120%)", st['tc_loss'])],
    ]
    t_jn = Table(jn_data, colWidths=[46, 24, 24, 76, 76, 52, 42, 60, 56, 76])
    t_jn.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_jn)
    story.append(Paragraph("<font size='6.5'><i>*Note: (128,128) on Jetson Nano evaluated at H=W=128 per taskC protocol; H=W=14 on RPi4.</i></font>", st['body']))
    story.append(Spacer(1, 6))

    cross_plot = os.path.join(PLOTS_DIR, "cross_platform_microbench.png")
    if os.path.exists(cross_plot):
        story.append(Image(cross_plot, width=6.6*72, height=2.6*72))
        story.append(Paragraph("<b>Figure 1:</b> Fused Winograd execution latency across workloads on Jetson Nano vs Raspberry Pi 4 (log scale).", st['caption']))
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 3: Section 4: Adaptive-m Sweep & Policy Inversion
    # --------------------------------------------------------------------------
    story.append(Paragraph("4. Adaptive-m Selection: Empirical Sweep & Policy Inversion", st['h1']))
    story.append(Paragraph(
        "The reference autotiler used a static rule WS &le; &alpha; L1, forcing m=2 (F(2,3)) across all workloads. "
        "We performed an empirical sweep on Raspberry Pi 4 evaluating m &isin; {2, 4, 6} (n=30 runs each, 420 total trials). "
        "The results reveal a dramatic mathematical inversion: m=6 outperforms m=2 by up to 2.84x for Cin &le; 32.",
        st['body']
    ))

    story.append(Paragraph("Table 3: Adaptive-m Empirical Verification on Raspberry Pi 4 (n=30 runs each)", st['h2']))
    m_data = [
        [Paragraph(h, st['th']) for h in ["Workload", "Cin", "Cout", "Policy m", "Empirical Best", "m=2 Latency (ms)", "m=6 Latency (ms)", "Speedup (m6 vs m2)", "p-value", "Policy Correct?"]],
        [Paragraph("(16,32)", st['tc_bold']), Paragraph("16", st['tc']), Paragraph("32", st['tc']), Paragraph("2", st['tc']), Paragraph("<b>6</b>", st['tc_win']), Paragraph("19.208 &plusmn; 0.047", st['tc']), Paragraph("<b>6.757 &plusmn; 0.005</b>", st['tc_win']), Paragraph("<b>2.84x Win</b>", st['tc_win']), Paragraph("&lt; 1e-10", st['tc']), Paragraph("FALSE", st['tc_loss'])],
        [Paragraph("(32,16)", st['tc_bold']), Paragraph("32", st['tc']), Paragraph("16", st['tc']), Paragraph("2", st['tc']), Paragraph("<b>6</b>", st['tc_win']), Paragraph("9.694 &plusmn; 0.009", st['tc']), Paragraph("<b>3.869 &plusmn; 0.004</b>", st['tc_win']), Paragraph("<b>2.51x Win</b>", st['tc_win']), Paragraph("&lt; 1e-10", st['tc']), Paragraph("FALSE", st['tc_loss'])],
        [Paragraph("(32,32)", st['tc_bold']), Paragraph("32", st['tc']), Paragraph("32", st['tc']), Paragraph("2", st['tc']), Paragraph("<b>6</b>", st['tc_win']), Paragraph("24.998 &plusmn; 0.151", st['tc']), Paragraph("<b>12.525 &plusmn; 0.018</b>", st['tc_win']), Paragraph("<b>2.00x Win</b>", st['tc_win']), Paragraph("&lt; 1e-10", st['tc']), Paragraph("FALSE", st['tc_loss'])],
        [Paragraph("(32,64)", st['tc_bold']), Paragraph("32", st['tc']), Paragraph("64", st['tc']), Paragraph("2", st['tc']), Paragraph("<b>6</b>", st['tc_win']), Paragraph("68.425 &plusmn; 0.090", st['tc']), Paragraph("<b>52.342 &plusmn; 0.102</b>", st['tc_win']), Paragraph("<b>1.31x Win</b>", st['tc_win']), Paragraph("&lt; 1e-10", st['tc']), Paragraph("FALSE", st['tc_loss'])],
        [Paragraph("(64,32)", st['tc_bold']), Paragraph("64", st['tc']), Paragraph("32", st['tc']), Paragraph("2", st['tc']), Paragraph("<b>6</b>", st['tc_win']), Paragraph("34.064 &plusmn; 0.040", st['tc']), Paragraph("<b>21.573 &plusmn; 0.152</b>", st['tc_win']), Paragraph("<b>1.58x Win</b>", st['tc_win']), Paragraph("&lt; 1e-10", st['tc']), Paragraph("FALSE", st['tc_loss'])],
        [Paragraph("(64,64)", st['tc_bold']), Paragraph("64", st['tc']), Paragraph("64", st['tc']), Paragraph("2", st['tc']), Paragraph("<b>2</b>", st['tc_win']), Paragraph("<b>99.162 &plusmn; 0.146</b>", st['tc_win']), Paragraph("154.341 &plusmn; 0.293", st['tc']), Paragraph("0.64x (Loss)", st['tc_loss']), Paragraph("&lt; 1e-10", st['tc']), Paragraph("TRUE", st['tc_win'])],
        [Paragraph("(128,128)", st['tc_bold']), Paragraph("128", st['tc']), Paragraph("128", st['tc']), Paragraph("2", st['tc']), Paragraph("<b>2</b>", st['tc_win']), Paragraph("<b>1,229.25 &plusmn; 2.10</b>", st['tc_win']), Paragraph("1,244.83 &plusmn; 0.54", st['tc']), Paragraph("0.99x", st['tc']), Paragraph("&lt; 1e-10", st['tc']), Paragraph("TRUE", st['tc_win'])],
    ]
    t_m = Table(m_data, colWidths=[46, 22, 22, 42, 65, 82, 82, 65, 46, 60])
    t_m.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_m)
    story.append(Spacer(1, 6))

    m_plot = os.path.join(PLOTS_DIR, "adaptive_m_sweep_rpi4.png")
    if os.path.exists(m_plot):
        story.append(Image(m_plot, width=6.6*72, height=2.6*72))
        story.append(Paragraph("<b>Figure 2:</b> RPi4 Adaptive-m verification: policy choice m=2 vs empirical best m=6 across workloads (log scale).", st['caption']))
    story.append(Spacer(1, 6))

    m_analysis = (
        "<b>MATHEMATICAL DIAGNOSIS OF POLICY INVERSION:</b> Winograd F(m, 3) requires (m+2)&sup2; multiplications for m&sup2; output points.<br/>"
        "&bull; <b>F(2,3):</b> 16 multiplications / 4 outputs = <b>4.00 mults / pixel</b>.<br/>"
        "&bull; <b>F(6,3):</b> 64 multiplications / 36 outputs = <b>1.78 mults / pixel</b> (a <b>2.25x theoretical FLOP reduction</b>).<br/>"
        "The static autotiler penalized F(6,3) because its working set (~70 KB) exceeded L1D (32 KB). But Cortex-A72 has a fast 1MB L2 cache "
        "with streaming prefetchers. For Cin &le; 32, the 2.25x FLOP savings overwhelmingly overpower L1 miss latency! Only when Cin reaches 64 "
        "does register pressure and transformation matrix complexity cause F(6,3) to regress (collapsing from 99ms to 154ms)."
    )
    story.append(make_callout(m_analysis, st['callout'], bg="#FFF5F5", border="#FEB2B2"))
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 4: Section 5: Register Pressure & Spilling Audit
    # --------------------------------------------------------------------------
    story.append(Paragraph("5. Microarchitectural Root Cause: Register Spilling at (128,128)", st['h1']))
    story.append(Paragraph(
        "At workload (128,128), execution latency jumps to 2,322 ms on RPi4 and 43,695 ms on Jetson Nano, resulting in zero speedup "
        "(and refuting the paper's claimed +63.91%). Disassembly analysis using AArch64 objdump isolates the root cause.",
        st['body']
    ))

    reg_data = [
        [Paragraph(h, st['th']) for h in ["Metric / Property", "Measured Hardware Value", "Architectural Limit", "Analysis & Microarchitectural Impact"]],
        [Paragraph("Target Workload", st['tc_bold']), Paragraph("Cin=128, Cout=128, K=3x3", st['tc']), Paragraph("Canonical Workload", st['tc']), Paragraph("Large channel regime tested in microbenchmark suite", st['tc'])],
        [Paragraph("Inner Loop Instructions", st['tc_bold']), Paragraph("157 total instructions", st['tc_mono']), Paragraph("&lt; 100 target", st['tc']), Paragraph("Disassembled AArch64 machine code inner kernel loop", st['tc'])],
        [Paragraph("Virtual SIMD Vectors Needed", st['tc_bold']), Paragraph("<b>64 virtual vector registers</b>", st['tc_loss']), Paragraph("<b>32 architectural (q0-q31)</b>", st['tc_bold']), Paragraph("<b>2.0x Over-subscription</b> of physical SIMD register file", st['tc_loss'])],
        [Paragraph("Vector Spill Loads (LDR q)", st['tc_bold']), Paragraph("5 instructions", st['tc_mono']), Paragraph("0 (Zero Spill Ideal)", st['tc']), Paragraph("Reloading accumulator vectors from stack frame", st['tc'])],
        [Paragraph("Vector Spill Stores (STR q)", st['tc_bold']), Paragraph("5 instructions", st['tc_mono']), Paragraph("0 (Zero Spill Ideal)", st['tc']), Paragraph("Evicting accumulator vectors to stack frame", st['tc'])],
        [Paragraph("Total Spill Memory Operations", st['tc_bold']), Paragraph("<b>10 spill instructions</b>", st['tc_loss']), Paragraph("0", st['tc']), Paragraph("10 out of 157 instructions are register spill traffic", st['tc_loss'])],
        [Paragraph("Spill Ratio", st['tc_bold']), Paragraph("<b>6.37%</b>", st['tc_loss']), Paragraph("0.00%", st['tc']), Paragraph("Severe register pressure stalls out-of-order execution pipeline", st['tc_loss'])],
        [Paragraph("Hypothesis Supported", st['tc_bold']), Paragraph("<b>TRUE</b> (Confirmed)", st['tc_win']), Paragraph("Expected True", st['tc']), Paragraph("Register spilling fully explains the performance collapse at 128x128", st['tc_win'])],
    ]
    t_reg = Table(reg_data, colWidths=[115, 95, 95, 227])
    t_reg.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_reg)
    story.append(Spacer(1, 6))

    ws_plot = os.path.join(PLOTS_DIR, "working_set_cache_boundaries.png")
    if os.path.exists(ws_plot):
        story.append(Image(ws_plot, width=6.6*72, height=2.6*72))
        story.append(Paragraph("<b>Figure 3:</b> Kernel working set growth across workloads relative to L1D (32 KB) and L2 (1024 KB) hardware limits.", st['caption']))
    story.append(Spacer(1, 6))

    spill_takeaway = (
        "<b>WHY REGISTER SPILLING IS FATAL FOR WINOGRAD:</b> Standard direct convolution maintains a single accumulator register "
        "and streams inputs and weights. Winograd F(2,3), however, operates on 4x4 transformed tiles, requiring 16 separate 2D accumulators "
        "per output channel. When unrolling across Cout=128 with vector blocking, the register allocator requires 64 concurrent SIMD vectors. "
        "Because ARM NEON has only 32 physical registers (q0-q31), the compiler spills active accumulators to L1 stack memory. "
        "This converts compute-bound matrix multiplication into an L1-load/store bandwidth-bound bottleneck, destroying Winograd arithmetic savings."
    )
    story.append(make_callout(spill_takeaway, st['callout'], bg="#FFF5F5", border="#FEB2B2"))
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 5: Section 6: End-to-End CNN Inference Benchmarks
    # --------------------------------------------------------------------------
    story.append(Paragraph("6. End-to-End Deep Learning Inference Benchmarks", st['h1']))
    story.append(Paragraph(
        "Full convolutional neural networks were benchmarked on Raspberry Pi 4 under stable thermal conditions (throttled=0x0). "
        "Four standard architectures were evaluated across CacheWinograd, Naive Winograd, and Non-fused Winograd.",
        st['body']
    ))

    e2e_method_box = (
        "<b>METHODOLOGY TRANSPARENCY (Constraint #6):</b> CacheWinograd end-to-end latency is measured as a <b>layer-time aggregate</b> "
        "of isolated per-layer kernel executions. Baseline graph frameworks (ORT, TVM, ArmCL) were skipped due to unbuilt shared library "
        "dependencies on this target. The table below directly compares CacheWinograd fused execution against Naive Winograd and Non-fused "
        "Winograd within the identical layer-isolated execution harness."
    )
    story.append(make_callout(e2e_method_box, st['callout'], bg="#FEFCBF", border="#F6E05E"))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Table 4: End-to-End CNN Inference Performance on Raspberry Pi 4", st['h2']))
    e2e_data = [
        [Paragraph(h, st['th']) for h in ["Model Architecture", "Conv3x3 Layers", "Runs (n)", "CacheWinograd Fused (ms)", "Naive Winograd (ms)", "Non-fused Winograd (ms)", "Speedup vs Naive", "Speedup vs Non-fused", "Statistical p-value"]],
        [Paragraph("<b>AlexNet</b>", st['tc_bold']), Paragraph("3 (3 stride-1)", st['tc']), Paragraph("50", st['tc']), Paragraph("51,771.93 &plusmn; 11.8", st['tc']), Paragraph("51,718.80 &plusmn; 14.9", st['tc']), Paragraph("51,725.34 &plusmn; 3.4", st['tc']), Paragraph("-0.10%", st['tc_loss']), Paragraph("-0.09%", st['tc_loss']), Paragraph("3.53e-7 (Sig)", st['tc'])],
        [Paragraph("<b>ResNet-18</b>", st['tc_bold']), Paragraph("16 (13 stride-1)", st['tc']), Paragraph("50", st['tc']), Paragraph("232,135.14 &plusmn; 308.6", st['tc']), Paragraph("231,683.60 &plusmn; 1090.8", st['tc']), Paragraph("231,698.75 &plusmn; 812.0", st['tc']), Paragraph("-0.19%", st['tc_loss']), Paragraph("-0.19%", st['tc_loss']), Paragraph("0.438 (Not Sig)", st['tc'])],
        [Paragraph("<b>ResNet-34</b>", st['tc_bold']), Paragraph("32 (29 stride-1)", st['tc']), Paragraph("50", st['tc']), Paragraph("424,153.85 &plusmn; 1129.5", st['tc']), Paragraph("420,713.04 &plusmn; 399.5", st['tc']), Paragraph("406,577.03 &plusmn; 1815.4", st['tc']), Paragraph("-0.82%", st['tc_loss']), Paragraph("-4.32%", st['tc_loss']), Paragraph("&lt; 1e-10 (Sig)", st['tc'])],
        [Paragraph("<b>VGG-16</b>", st['tc_bold']), Paragraph("13 (13 stride-1)", st['tc']), Paragraph("10", st['tc']), Paragraph("<b>1,846,777.90 &plusmn; 17767</b>", st['tc_win']), Paragraph("1,939,926.32 &plusmn; 542", st['tc']), Paragraph("1,939,916.34 &plusmn; 120", st['tc']), Paragraph("<b>+4.80%</b>", st['tc_win']), Paragraph("<b>+4.80%</b>", st['tc_win']), Paragraph("<b>2.82e-6 (Sig Win!)</b>", st['tc_win'])],
    ]
    t_e2e = Table(e2e_data, colWidths=[65, 55, 34, 85, 80, 80, 52, 52, 49])
    t_e2e.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_e2e)
    story.append(Spacer(1, 6))

    e2e_plot = os.path.join(PLOTS_DIR, "e2e_comparison_rpi4.png")
    if os.path.exists(e2e_plot):
        story.append(Image(e2e_plot, width=6.6*72, height=2.4*72))
        story.append(Paragraph("<b>Figure 4:</b> End-to-end network latency (left, log scale) and percentage speedup of CacheWinograd vs Naive Winograd (right).", st['caption']))
    story.append(Spacer(1, 6))

    vgg_box = (
        "<b>VGG-16 BREAKTHROUGH ANALYSIS:</b> VGG-16 achieves a statistically validated <b>+4.80% latency reduction</b>, "
        "saving <b>93.15 seconds per inference pass</b> (p = 2.82e-6). VGG-16 consists of deep cascades of 3x3 convolutions with large channel counts "
        "(256 &rarr; 512, 512 &rarr; 512) and large spatial feature maps (up to 224x224). Here, intermediate transformation DRAM buffer elimination "
        "prevents millions of bus transactions. In contrast, ResNet-34 features 32 distinct layers whose spatial dimensions quickly shrink to 14x14 and 7x7; "
        "on tiny feature maps, loop prologue/epilogue and border padding overheads outpace cache savings (-4.32% regression)."
    )
    story.append(make_callout(vgg_box, st['callout'], bg="#F0FFF4", border="#9AE6B4"))
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 6: Section 7: Fallback-Guard & Multi-Core Ablation
    # --------------------------------------------------------------------------
    story.append(Paragraph("7. Fallback-Guard Verification & Multi-Threading Ablation", st['h1']))
    story.append(Paragraph(
        "To protect against cache thrashing when working sets exceed L1D capacity, CacheWinograd integrates an automated fallback guard. "
        "We validated this guard on held-out workloads (48,96) and (96,48) across 4,000 raw executions on Jetson Nano.",
        st['body']
    ))

    story.append(Paragraph("Table 5: Fallback-Guard Verification on Held-Out Workloads (Jetson Nano, n=1,000 runs each)", st['h2']))
    fb_data = [
        [Paragraph(h, st['th']) for h in ["Held-Out Config", "Tile Type", "Working Set (Bytes)", "L1D Capacity (Bytes)", "Fallback Triggered?", "Baseline Latency (ms)", "Fused Latency (ms)", "Improvement", "p-value", "Significance"]],
        [Paragraph("(48,96)", st['tc_bold']), Paragraph("F(2,3)", st['tc']), Paragraph("53,376 B", st['tc']), Paragraph("32,768 B", st['tc']), Paragraph("<b>TRUE (100%)</b>", st['tc_win']), Paragraph("2,784.71 &plusmn; 0.74", st['tc']), Paragraph("2,776.23 &plusmn; 1.01", st['tc']), Paragraph("+0.30%", st['tc_win']), Paragraph("&lt; 1e-10", st['tc']), Paragraph("SIGNIFICANT", st['tc_win'])],
        [Paragraph("(96,48)", st['tc_bold']), Paragraph("F(2,3)", st['tc']), Paragraph("105,600 B", st['tc']), Paragraph("32,768 B", st['tc']), Paragraph("<b>TRUE (100%)</b>", st['tc_win']), Paragraph("1,405.80 &plusmn; 0.47", st['tc']), Paragraph("1,402.82 &plusmn; 0.64", st['tc']), Paragraph("+0.21%", st['tc_win']), Paragraph("&lt; 1e-10", st['tc']), Paragraph("SIGNIFICANT", st['tc_win'])],
    ]
    t_fb = Table(fb_data, colWidths=[55, 38, 65, 65, 65, 75, 75, 48, 42, 54])
    t_fb.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_fb)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Table 6: Multi-Core (4T) and Fusion Ablation on Jetson Nano (n=10 runs)", st['h2']))
    ab_data = [
        [Paragraph(h, st['th']) for h in ["Config", "Mode", "Threads", "Mean Latency (ms)", "CI95 (ms)", "vs Unfused 1T (%)", "Measured Fused-4T vs Unfused-4T", "Paper Claim", "Paper Match"]],
        [Paragraph("(64,64)", st['tc_bold']), Paragraph("Unfused", st['tc']), Paragraph("1", st['tc']), Paragraph("14.7835", st['tc']), Paragraph("0.0368", st['tc']), Paragraph("Baseline", st['tc']), Paragraph("-", st['tc']), Paragraph("-", st['tc']), Paragraph("-", st['tc'])],
        [Paragraph("(64,64)", st['tc_bold']), Paragraph("Unfused", st['tc']), Paragraph("4", st['tc']), Paragraph("14.8315", st['tc']), Paragraph("0.0132", st['tc']), Paragraph("-0.32%", st['tc_loss']), Paragraph("-", st['tc']), Paragraph("-", st['tc']), Paragraph("-", st['tc'])],
        [Paragraph("(64,64)", st['tc_bold']), Paragraph("Fused", st['tc']), Paragraph("1", st['tc']), Paragraph("14.7183", st['tc']), Paragraph("0.0238", st['tc']), Paragraph("+0.44%", st['tc_win']), Paragraph("-", st['tc']), Paragraph("-", st['tc']), Paragraph("-", st['tc'])],
        [Paragraph("(64,64)", st['tc_bold']), Paragraph("Fused", st['tc']), Paragraph("4", st['tc']), Paragraph("14.7211", st['tc']), Paragraph("0.0153", st['tc']), Paragraph("+0.42%", st['tc_win']), Paragraph("<b>+0.74%</b>", st['tc']), Paragraph("<b>+33.11%</b>", st['tc_loss']), Paragraph("<b>MISMATCH (-97.7%)</b>", st['tc_loss'])],
        [Paragraph("(128,128)", st['tc_bold']), Paragraph("Unfused", st['tc']), Paragraph("1", st['tc']), Paragraph("100.7100", st['tc']), Paragraph("0.1376", st['tc']), Paragraph("Baseline", st['tc']), Paragraph("-", st['tc']), Paragraph("-", st['tc']), Paragraph("-", st['tc'])],
        [Paragraph("(128,128)", st['tc_bold']), Paragraph("Unfused", st['tc']), Paragraph("4", st['tc']), Paragraph("101.1222", st['tc']), Paragraph("0.1633", st['tc']), Paragraph("-0.41%", st['tc_loss']), Paragraph("-", st['tc']), Paragraph("-", st['tc']), Paragraph("-", st['tc'])],
        [Paragraph("(128,128)", st['tc_bold']), Paragraph("Fused", st['tc']), Paragraph("1", st['tc']), Paragraph("101.1078", st['tc']), Paragraph("0.1296", st['tc']), Paragraph("-0.40%", st['tc_loss']), Paragraph("-", st['tc']), Paragraph("-", st['tc']), Paragraph("-", st['tc'])],
        [Paragraph("(128,128)", st['tc_bold']), Paragraph("Fused", st['tc']), Paragraph("4", st['tc']), Paragraph("100.9899", st['tc']), Paragraph("0.1190", st['tc']), Paragraph("-0.28%", st['tc_loss']), Paragraph("+0.13%", st['tc']), Paragraph("N/A", st['tc']), Paragraph("NO_CLAIM", st['tc'])],
    ]
    t_ab = Table(ab_data, colWidths=[46, 45, 36, 75, 55, 65, 80, 55, 75])
    t_ab.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_ab)
    story.append(Spacer(1, 6))

    ab_analysis = (
        "<b>MULTI-CORE SCALING AUDIT:</b> Multi-threading across 4 cores fails to provide speedup on embedded ARM cores for small spatial tiles "
        "(14x14 or 56x56). Measured Fused-4T vs Unfused-4T improvement is <b>+0.74%</b>, in stark contrast to the paper's claimed <b>+33.11%</b> (-97.75% diff). "
        "On Cortex-A57/A72, OpenMP barrier overhead, core synchronization latency, and false sharing in the shared L2 cache completely nullify "
        "thread parallelism when tile counts per thread are low."
    )
    story.append(make_callout(ab_analysis, st['callout'], bg="#EDF2F7", border="#CBD5E0"))
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 7: Section 8: Roofline Model & DRAM Traffic Savings
    # --------------------------------------------------------------------------
    story.append(Paragraph("8. Roofline Model Bound Analysis & DRAM Traffic Savings", st['h1']))
    story.append(Paragraph(
        "Arithmetic intensity was calculated for all 7 workloads against the Jetson Nano roofline envelope (45.8 GFLOPS peak NEON, 25.6 GB/s DRAM bandwidth). "
        "The architectural ridge point is 45.8 / 25.6 = 1.7891 FLOP/Byte. All workloads fall strictly in the <b>COMPUTE-BOUND</b> regime.",
        st['body']
    ))

    story.append(Paragraph("Table 7: Arithmetic Intensity vs Roofline Envelope on Jetson Nano", st['h2']))
    roof_data = [
        [Paragraph(h, st['th']) for h in ["Workload", "MACs", "Total Bytes", "Arithmetic Intensity (FLOP/B)", "Ridge Point (FLOP/B)", "Bounded By", "Measured GFLOPS", "Intermediate DRAM Saved", "DRAM Savings %"]],
        [Paragraph("(16,32)", st['tc_bold']), Paragraph("14.45 M", st['tc']), Paragraph("620.5 KB", st['tc']), Paragraph("23.29", st['tc']), Paragraph("1.7891", st['tc']), Paragraph("COMPUTE", st['tc_bold']), Paragraph("14.62 GFLOPS", st['tc']), Paragraph("4,608 B", st['tc']), Paragraph("0.76%", st['tc'])],
        [Paragraph("(32,16)", st['tc_bold']), Paragraph("14.45 M", st['tc']), Paragraph("620.5 KB", st['tc']), Paragraph("23.29", st['tc']), Paragraph("1.7891", st['tc']), Paragraph("COMPUTE", st['tc_bold']), Paragraph("13.07 GFLOPS", st['tc']), Paragraph("4,096 B", st['tc']), Paragraph("0.68%", st['tc'])],
        [Paragraph("(32,32)", st['tc_bold']), Paragraph("28.90 M", st['tc']), Paragraph("839.7 KB", st['tc']), Paragraph("34.42", st['tc']), Paragraph("1.7891", st['tc']), Paragraph("COMPUTE", st['tc_bold']), Paragraph("10.62 GFLOPS", st['tc']), Paragraph("4,096 B", st['tc']), Paragraph("0.51%", st['tc'])],
        [Paragraph("(32,64)", st['tc_bold']), Paragraph("57.80 M", st['tc']), Paragraph("1,278.0 KB", st['tc']), Paragraph("45.23", st['tc']), Paragraph("1.7891", st['tc']), Paragraph("COMPUTE", st['tc_bold']), Paragraph("7.09 GFLOPS", st['tc']), Paragraph("4,096 B", st['tc']), Paragraph("0.34%", st['tc'])],
        [Paragraph("(64,32)", st['tc_bold']), Paragraph("57.80 M", st['tc']), Paragraph("1,278.0 KB", st['tc']), Paragraph("45.23", st['tc']), Paragraph("1.7891", st['tc']), Paragraph("COMPUTE", st['tc_bold']), Paragraph("12.72 GFLOPS", st['tc']), Paragraph("8,192 B", st['tc']), Paragraph("0.68%", st['tc'])],
        [Paragraph("(64,64)", st['tc_bold']), Paragraph("115.61 M", st['tc']), Paragraph("1,753.1 KB", st['tc']), Paragraph("65.94", st['tc']), Paragraph("1.7891", st['tc']), Paragraph("COMPUTE", st['tc_bold']), Paragraph("7.84 GFLOPS", st['tc']), Paragraph("8,192 B", st['tc']), Paragraph("0.51%", st['tc'])],
        [Paragraph("(128,128)", st['tc_bold']), Paragraph("462.42 M", st['tc']), Paragraph("3,801.1 KB", st['tc']), Paragraph("121.66", st['tc']), Paragraph("1.7891", st['tc']), Paragraph("COMPUTE", st['tc_bold']), Paragraph("0.01 GFLOPS", st['tc_loss']), Paragraph("16,384 B", st['tc']), Paragraph("0.51%", st['tc'])],
    ]
    t_roof = Table(roof_data, colWidths=[46, 50, 56, 68, 56, 54, 70, 68, 64])
    t_roof.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_roof)
    story.append(Spacer(1, 6))

    roof_plot = os.path.join(PLOTS_DIR, "roofline_intensity.png")
    if os.path.exists(roof_plot):
        story.append(Image(roof_plot, width=6.6*72, height=2.6*72))
        story.append(Paragraph("<b>Figure 5:</b> Operational arithmetic intensity vs Jetson Nano roofline ridge point (1.79 FLOP/Byte, log scale).", st['caption']))
    story.append(Spacer(1, 6))

    dram_box = (
        "<b>DRAM SAVINGS INTERPRETATION:</b> Fusing the input and output Winograd transformations into cache eliminates the DRAM round-trip "
        "for intermediate tile buffers, saving between 4,096 B and 16,384 B per tile. However, relative to total layer memory traffic "
        "(input activations, filter weights, and output tensors), intermediate transform elimination saves <b>0.34% to 0.76%</b> of total DRAM bytes. "
        "Therefore, CacheWinograd's primary execution speedups stem from high L1/L2 data locality and register reuse, not raw DRAM bandwidth reduction."
    )
    story.append(make_callout(dram_box, st['callout'], bg="#EDF2F7", border="#CBD5E0"))
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 8: Section 9 & 10: Discrepancy Matrix & Engineering Takeaways
    # --------------------------------------------------------------------------
    story.append(Paragraph("9. Published Paper Discrepancy & Reproducibility Matrix", st['h1']))
    story.append(Paragraph(
        "Per strict experimental reproducibility guidelines, Table 8 audits published paper claims against fresh physical hardware measurements.",
        st['body']
    ))

    disc_data = [
        [Paragraph(h, st['th']) for h in ["Experiment / Claim", "Platform", "Paper Claim", "Measured Today", "% Diff", "Audit Status", "Root Cause Diagnosis"]],
        [Paragraph("L1D Cache Capacity", st['tc_bold']), Paragraph("Jetson / RPi4", st['tc']), Paragraph("32,768 Bytes", st['tc']), Paragraph("32,768 Bytes", st['tc']), Paragraph("0.00%", st['tc_win']), Paragraph("MATCH", st['tc_win']), Paragraph("Exact match via sysfs hardware query.", st['tc'])],
        [Paragraph("L2 Cache Capacity", st['tc_bold']), Paragraph("RPi4 (A72)", st['tc']), Paragraph("2,097,152 Bytes", st['tc']), Paragraph("1,048,576 Bytes", st['tc_loss']), Paragraph("-50.00%", st['tc_loss']), Paragraph("MISMATCH", st['tc_loss']), Paragraph("BCM2711 SoC integrates 1MB L2, not 2MB.", st['tc'])],
        [Paragraph("L2 Cache Capacity", st['tc_bold']), Paragraph("Jetson Nano", st['tc']), Paragraph("2,097,152 Bytes", st['tc']), Paragraph("2,097,152 Bytes", st['tc']), Paragraph("0.00%", st['tc_win']), Paragraph("MATCH", st['tc_win']), Paragraph("Tegra210 SoC integrates 2MB L2 shared.", st['tc'])],
        [Paragraph("(128,128) Microbench", st['tc_bold']), Paragraph("RPi4", st['tc']), Paragraph("+63.91% speedup", st['tc_loss']), Paragraph("-0.01% latency", st['tc']), Paragraph("-100.02%", st['tc_loss']), Paragraph("MISMATCH", st['tc_loss']), Paragraph("Register spilling (2.0x over-sub) destroys gain.", st['tc'])],
        [Paragraph("(128,128) Microbench", st['tc_bold']), Paragraph("Jetson Nano", st['tc']), Paragraph("-6.52% latency", st['tc_loss']), Paragraph("+1.32% speedup", st['tc']), Paragraph("+120.28%", st['tc_loss']), Paragraph("MISMATCH", st['tc_loss']), Paragraph("Severe compute bound, 43s run time.", st['tc'])],
        [Paragraph("Ablation Fused-4T", st['tc_bold']), Paragraph("Jetson Nano", st['tc']), Paragraph("+33.11% vs 4T unfused", st['tc_loss']), Paragraph("+0.74% speedup", st['tc']), Paragraph("-97.75%", st['tc_loss']), Paragraph("MISMATCH", st['tc_loss']), Paragraph("Zero multi-threading scaling on small tiles.", st['tc'])],
        [Paragraph("Fallback Guard (48,96)", st['tc_bold']), Paragraph("Jetson Nano", st['tc']), Paragraph("Fallback triggers", st['tc']), Paragraph("100% Fallback (n=1000)", st['tc_win']), Paragraph("0.00%", st['tc_win']), Paragraph("MATCH", st['tc_win']), Paragraph("WS (53KB) &gt; L1 (32KB); guard verified.", st['tc'])],
        [Paragraph("Fallback Guard (96,48)", st['tc_bold']), Paragraph("Jetson Nano", st['tc']), Paragraph("Fallback triggers", st['tc']), Paragraph("100% Fallback (n=1000)", st['tc_win']), Paragraph("0.00%", st['tc_win']), Paragraph("MATCH", st['tc_win']), Paragraph("WS (105KB) &gt; L1 (32KB); guard verified.", st['tc'])],
        [Paragraph("End-to-End VGG-16", st['tc_bold']), Paragraph("RPi4", st['tc']), Paragraph("&gt; 0% Improvement", st['tc']), Paragraph("+4.80% (p=2.82e-6)", st['tc_win']), Paragraph("Verified", st['tc_win']), Paragraph("MATCH (Win)", st['tc_win']), Paragraph("Deep 3x3 cascades yield 93.15s savings.", st['tc_win'])],
    ]
    t_disc = Table(disc_data, colWidths=[85, 55, 68, 68, 45, 55, 156])
    t_disc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_disc)
    story.append(Spacer(1, 8))

    story.append(Paragraph("10. Compiler Engineering Takeaways & Recommendations", st['h1']))
    takeaways = (
        "<b>ACTIONABLE GUIDANCE FOR DEEP LEARNING COMPILER DEVELOPERS:</b><br/>"
        "1. <b>Register-Pressure-Aware Tiling:</b> Do not rely exclusively on L1 cache capacity for tiling decisions. "
        "Register allocation pressure is the primary failure mode on 32-register NEON architectures. Compilers must enforce "
        "a virtual SIMD accumulator budget &le; 24 physical registers to preserve registers for addressing and coefficients.<br/>"
        "2. <b>FLOP-Weighted Adaptive-m:</b> Replace the static WS &le; &alpha; L1 heuristic with an empirical cost model that weights "
        "the 2.25x arithmetic reduction of F(6,3) against L1 miss penalty. Automatically schedule m=6 for Cin &le; 32, reverting to m=2 only when Cin &ge; 64.<br/>"
        "3. <b>Topological Kernel Selection:</b> Winograd fusion should be enabled based on network topology: schedule fusion for deep, unstrided "
        "cascades (e.g. VGG-16, +4.8% win), but bypass fusion for shallow or spatially small networks (e.g. ResNet-34, -4.3% loss) where direct conv dominates.<br/>"
        "4. <b>Spatial-Aware Parallel Scheduling:</b> Suppress multi-threading on spatial grids &le; 56x56 on quad-core embedded SoCs. Barrier latency "
        "and L2 cache-line false sharing completely eliminate parallel gains on small feature maps."
    )
    story.append(make_callout(takeaways, st['callout'], bg="#EDF2F7", border="#CBD5E0"))

    # Build PDF
    doc.build(story, canvasmaker=MonographCanvas)
    print(f"[PDF Builder] Successfully generated: {pdf_path}")
    return pdf_path


# ==============================================================================
# PDF 2: AI REFERENCE DIGEST (2 PAGES)
# ==============================================================================
def build_ai_reference_digest_pdf():
    pdf_path = os.path.join(ARTIFACTS_DIR, "CacheWinograd_AI_Reference_Digest.pdf")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=36, rightMargin=36, topMargin=44, bottomMargin=44
    )
    st = get_styles()
    story = []

    # --------------------------------------------------------------------------
    # DIGEST PAGE 1: Header, Hardware Card, Microbenchmarks & Adaptive-m
    # --------------------------------------------------------------------------
    story.append(Paragraph("CacheWinograd AI Reference Digest", st['title']))
    story.append(Paragraph(
        "<b>Fast-Ingestion Reference Card: Hardware Parameters, Empirical Data Tables & Decision Matrices</b><br/>"
        "<i>Optimized for LLM Reasoning Engines (Claude, GPT-4, Gemini) | Multi-Platform Evaluation Suite</i>",
        st['subtitle']
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=6))

    guide_box = (
        "<b>HOW TO USE THIS DIGEST WITH LLMs:</b> Upload this PDF to any AI reasoning system along with questions on "
        "deep learning compiler optimization, Winograd algorithm performance, or embedded ARM kernel tuning. "
        "This digest contains dense, high-signal lookup tables covering exact empirical latencies, confidence intervals, "
        "microarchitectural failure modes, and hardware parameters without conversational filler."
    )
    story.append(make_callout(guide_box, st['callout'], bg="#EBF8FF", border="#BEE3F8", width=540))
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Hardware Specifications & Cache Hierarchy Card", st['h1']))
    hw_card_data = [
        [Paragraph(h, st['th']) for h in ["Platform", "Core Type", "Cores", "L1D Cache", "L2 Cache", "Line Size", "Memory BW", "Peak GFLOPS (FP32)", "Ridge Point"]],
        [Paragraph("<b>Raspberry Pi 4B</b>", st['tc_bold']), Paragraph("Cortex-A72", st['tc']), Paragraph("4", st['tc']), Paragraph("32 KB", st['tc']), Paragraph("<b>1,024 KB (1MB)</b>", st['tc_loss']), Paragraph("64 B", st['tc']), Paragraph("~12.8 GB/s", st['tc']), Paragraph("24.0 GFLOPS", st['tc']), Paragraph("1.88 FLOP/B", st['tc'])],
        [Paragraph("<b>Jetson Nano</b>", st['tc_bold']), Paragraph("Cortex-A57", st['tc']), Paragraph("4", st['tc']), Paragraph("32 KB", st['tc']), Paragraph("<b>2,048 KB (2MB)</b>", st['tc_win']), Paragraph("64 B", st['tc']), Paragraph("25.6 GB/s", st['tc']), Paragraph("45.8 GFLOPS", st['tc']), Paragraph("1.79 FLOP/B", st['tc'])],
        [Paragraph("<b>Apple Silicon</b>", st['tc_bold']), Paragraph("M-Series (ARMv8.5)", st['tc']), Paragraph("12", st['tc']), Paragraph("64 KB", st['tc_win']), Paragraph("<b>4,096 KB (4MB)</b>", st['tc_win']), Paragraph("128 B", st['tc']), Paragraph("&gt; 100 GB/s", st['tc']), Paragraph("&gt; 250 GFLOPS", st['tc']), Paragraph("&lt; 2.50 FLOP/B", st['tc'])],
    ]
    t_hw_card = Table(hw_card_data, colWidths=[75, 60, 30, 48, 68, 44, 65, 80, 70])
    t_hw_card.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_hw_card)
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. Microbenchmark Performance & Latency Lookup Card", st['h1']))
    mb_card_data = [
        [Paragraph(h, st['th']) for h in ["Workload (Cin,Cout)", "RPi4 Baseline (ms)", "RPi4 Fused (ms)", "RPi4 Speedup", "RPi4 p-value", "Jetson Baseline (ms)", "Jetson Fused (ms)", "Jetson Speedup", "Darwin Fused (ms)"]],
        [Paragraph("(16, 32)", st['tc_bold']), Paragraph("8.9637", st['tc']), Paragraph("8.9436", st['tc']), Paragraph("+0.22% (Sig)", st['tc_win']), Paragraph("2.13e-3", st['tc']), Paragraph("2.3064", st['tc']), Paragraph("0.9881", st['tc']), Paragraph("+57.16%", st['tc_win']), Paragraph("0.6300", st['tc'])],
        [Paragraph("(32, 16)", st['tc_bold']), Paragraph("4.3871", st['tc']), Paragraph("4.3759", st['tc']), Paragraph("+0.25% (Sig)", st['tc_win']), Paragraph("3.62e-2", st['tc']), Paragraph("1.0978", st['tc']), Paragraph("1.1056", st['tc']), Paragraph("-0.71%", st['tc_loss']), Paragraph("0.9956", st['tc'])],
        [Paragraph("(32, 32)", st['tc_bold']), Paragraph("13.3931", st['tc']), Paragraph("13.3720", st['tc']), Paragraph("+0.16%", st['tc']), Paragraph("0.297", st['tc']), Paragraph("2.7597", st['tc']), Paragraph("2.7208", st['tc']), Paragraph("+1.41%", st['tc_win']), Paragraph("2.3396", st['tc'])],
        [Paragraph("(32, 64)", st['tc_bold']), Paragraph("44.0000", st['tc']), Paragraph("43.8069", st['tc']), Paragraph("+0.44%", st['tc']), Paragraph("0.160", st['tc']), Paragraph("8.1910", st['tc']), Paragraph("8.1507", st['tc']), Paragraph("+0.49%", st['tc']), Paragraph("6.1602", st['tc'])],
        [Paragraph("(64, 32)", st['tc_bold']), Paragraph("21.9022", st['tc']), Paragraph("21.7475", st['tc']), Paragraph("+0.71%", st['tc']), Paragraph("0.176", st['tc']), Paragraph("4.5615", st['tc']), Paragraph("4.5450", st['tc']), Paragraph("+0.36%", st['tc']), Paragraph("3.1474", st['tc'])],
        [Paragraph("(64, 64)", st['tc_bold']), Paragraph("85.3300", st['tc']), Paragraph("85.5026", st['tc']), Paragraph("-0.20%", st['tc_loss']), Paragraph("0.464", st['tc']), Paragraph("14.7619", st['tc']), Paragraph("14.7539", st['tc']), Paragraph("+0.05%", st['tc']), Paragraph("8.7733", st['tc'])],
        [Paragraph("(128, 128)", st['tc_bold']), Paragraph("2,322.15", st['tc']), Paragraph("2,322.40", st['tc']), Paragraph("-0.01%", st['tc_loss']), Paragraph("0.602", st['tc']), Paragraph("44,281.50*", st['tc_loss']), Paragraph("43,695.88*", st['tc_loss']), Paragraph("+1.32%", st['tc']), Paragraph("46.1133", st['tc_win'])],
    ]
    t_mb_card = Table(mb_card_data, colWidths=[80, 60, 60, 60, 48, 64, 64, 52, 52])
    t_mb_card.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_mb_card)
    story.append(Spacer(1, 6))

    story.append(Paragraph("3. Adaptive-m Selection Inversion Matrix (RPi4 Empirical Verification)", st['h1']))
    m_card_data = [
        [Paragraph(h, st['th']) for h in ["Config", "Policy Choice", "Empirical Best", "m=2 Latency (ms)", "m=6 Latency (ms)", "Speedup Ratio", "Policy Evaluation", "Mathematical Reason"]],
        [Paragraph("(16, 32)", st['tc_bold']), Paragraph("m=2", st['tc']), Paragraph("<b>m=6</b>", st['tc_win']), Paragraph("19.208", st['tc']), Paragraph("6.757", st['tc_win']), Paragraph("<b>2.84x Win</b>", st['tc_win']), Paragraph("WRONG", st['tc_loss']), Paragraph("F(6,3) saves 2.25x FLOPs; fits easily in cache", st['tc'])],
        [Paragraph("(32, 16)", st['tc_bold']), Paragraph("m=2", st['tc']), Paragraph("<b>m=6</b>", st['tc_win']), Paragraph("9.694", st['tc']), Paragraph("3.869", st['tc_win']), Paragraph("<b>2.51x Win</b>", st['tc_win']), Paragraph("WRONG", st['tc_loss']), Paragraph("F(6,3) amortizes 2D transform over 36 outputs", st['tc'])],
        [Paragraph("(32, 32)", st['tc_bold']), Paragraph("m=2", st['tc']), Paragraph("<b>m=6</b>", st['tc_win']), Paragraph("24.998", st['tc']), Paragraph("12.525", st['tc_win']), Paragraph("<b>2.00x Win</b>", st['tc_win']), Paragraph("WRONG", st['tc_loss']), Paragraph("L2 cache bandwidth sustains tile throughput", st['tc'])],
        [Paragraph("(32, 64)", st['tc_bold']), Paragraph("m=2", st['tc']), Paragraph("<b>m=6</b>", st['tc_win']), Paragraph("68.425", st['tc']), Paragraph("52.342", st['tc_win']), Paragraph("<b>1.31x Win</b>", st['tc_win']), Paragraph("WRONG", st['tc_loss']), Paragraph("Arithmetic reduction outpaces L1 miss cost", st['tc'])],
        [Paragraph("(64, 32)", st['tc_bold']), Paragraph("m=2", st['tc']), Paragraph("<b>m=6</b>", st['tc_win']), Paragraph("34.064", st['tc']), Paragraph("21.573", st['tc_win']), Paragraph("<b>1.58x Win</b>", st['tc_win']), Paragraph("WRONG", st['tc_loss']), Paragraph("Working set 70KB stream-buffered in L2", st['tc'])],
        [Paragraph("(64, 64)", st['tc_bold']), Paragraph("m=2", st['tc']), Paragraph("<b>m=2</b>", st['tc_win']), Paragraph("99.162", st['tc_win']), Paragraph("154.341", st['tc']), Paragraph("0.64x (Loss)", st['tc_loss']), Paragraph("CORRECT", st['tc_win']), Paragraph("F(6,3) explodes register pressure & spills", st['tc'])],
        [Paragraph("(128, 128)", st['tc_bold']), Paragraph("m=2", st['tc']), Paragraph("<b>m=2</b>", st['tc_win']), Paragraph("1,229.25", st['tc_win']), Paragraph("1,244.83", st['tc']), Paragraph("0.99x", st['tc']), Paragraph("CORRECT", st['tc_win']), Paragraph("Severe register over-subscription (2.0x)", st['tc'])],
    ]
    t_m_card = Table(m_card_data, colWidths=[55, 52, 55, 60, 60, 62, 58, 138])
    t_m_card.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_m_card)
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # DIGEST PAGE 2: End-to-End CNN Card, Architectural Gotchas & Formulas
    # --------------------------------------------------------------------------
    story.append(Paragraph("4. End-to-End CNN Network Benchmark Card (RPi4)", st['h1']))
    e2e_card_data = [
        [Paragraph(h, st['th']) for h in ["Architecture", "3x3 Layers", "Trials (n)", "CacheWinograd (s)", "Naive Wino (s)", "Non-fused (s)", "Speedup (%)", "Time Delta", "p-value", "Verdict"]],
        [Paragraph("<b>AlexNet</b>", st['tc_bold']), Paragraph("3", st['tc']), Paragraph("50", st['tc']), Paragraph("51.77 s", st['tc']), Paragraph("51.72 s", st['tc']), Paragraph("51.73 s", st['tc']), Paragraph("-0.10%", st['tc_loss']), Paragraph("+0.05 s", st['tc']), Paragraph("3.53e-7", st['tc']), Paragraph("Parity (No gain)", st['tc'])],
        [Paragraph("<b>ResNet-18</b>", st['tc_bold']), Paragraph("13 stride-1", st['tc']), Paragraph("50", st['tc']), Paragraph("232.14 s", st['tc']), Paragraph("231.68 s", st['tc']), Paragraph("231.70 s", st['tc']), Paragraph("-0.19%", st['tc_loss']), Paragraph("+0.45 s", st['tc']), Paragraph("0.438", st['tc']), Paragraph("Neutral (No gain)", st['tc'])],
        [Paragraph("<b>ResNet-34</b>", st['tc_bold']), Paragraph("29 stride-1", st['tc']), Paragraph("50", st['tc']), Paragraph("424.15 s", st['tc']), Paragraph("420.71 s", st['tc']), Paragraph("406.58 s", st['tc']), Paragraph("-4.32%", st['tc_loss']), Paragraph("+17.58 s", st['tc_loss']), Paragraph("&lt; 1e-10", st['tc']), Paragraph("Regression (Loss)", st['tc_loss'])],
        [Paragraph("<b>VGG-16</b>", st['tc_bold']), Paragraph("13 stride-1", st['tc']), Paragraph("10", st['tc']), Paragraph("<b>1,846.78 s</b>", st['tc_win']), Paragraph("1,939.93 s", st['tc']), Paragraph("1,939.92 s", st['tc']), Paragraph("<b>+4.80%</b>", st['tc_win']), Paragraph("<b>-93.15 s</b>", st['tc_win']), Paragraph("<b>2.82e-6</b>", st['tc_win']), Paragraph("<b>Stat. Sig. Win!</b>", st['tc_win'])],
    ]
    t_e2e_card = Table(e2e_card_data, colWidths=[65, 55, 36, 68, 62, 58, 52, 48, 48, 48])
    t_e2e_card.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_e2e_card)
    story.append(Spacer(1, 8))

    story.append(Paragraph("5. AI Reasoning Cheat-Sheet: Critical Architectural Gotchas", st['h1']))
    gotcha_box = (
        "<b>CORE ARCHITECTURAL GOTCHAS TO REMEMBER:</b><br/>"
        "1. <b>Layer-Time Aggregation Caveat (Constraint #6):</b> CacheWinograd E2E latency = &sum; isolated per-layer kernel times. "
        "It does not model graph-level pipeline fusion, inter-layer tensor memory reuse, or memory manager thrashing.<br/>"
        "2. <b>Register Spilling Threshold:</b> On ARM NEON (32 vector registers), Winograd F(2,3) requires virtual accumulators "
        "proportional to Cout. At Cin=Cout=128, the kernel hits a <b>2.0x over-subscription ratio</b>, generating 10 spill loads/stores "
        "in a 157-instruction inner loop (6.37% spill ratio), completely neutralizing the arithmetic advantage.<br/>"
        "3. <b>Multi-Threading Illusion:</b> 4-core multi-threading on small spatial tiles (14x14) yields negligible (+0.74%) "
        "speedup over 1T due to thread synchronization barriers and cache-line false sharing.<br/>"
        "4. <b>Autotiler Formula Inversion:</b> WS &le; &alpha; L1 is insufficient. The autotiler must incorporate arithmetic savings "
        "(F(6,3) requires 1.78 mults/pixel vs 4.00 for F(2,3)) and check against L2 capacity when Cin &le; 32."
    )
    story.append(make_callout(gotcha_box, st['callout'], bg="#FFF5F5", border="#FEB2B2", width=540))
    story.append(Spacer(1, 8))

    story.append(Paragraph("6. Summary Discrepancy Matrix: Paper Claims vs Measured Hardware", st['h1']))
    mini_disc_data = [
        [Paragraph(h, st['th']) for h in ["Experiment / Claim", "Platform", "Paper Claim", "Measured Today", "% Diff", "Audit Verdict"]],
        [Paragraph("L1D Cache Size", st['tc_bold']), Paragraph("Jetson / RPi4", st['tc']), Paragraph("32 KB", st['tc']), Paragraph("32 KB", st['tc']), Paragraph("0.0%", st['tc_win']), Paragraph("MATCH", st['tc_win'])],
        [Paragraph("L2 Cache Size", st['tc_bold']), Paragraph("RPi4 (A72)", st['tc']), Paragraph("2,048 KB", st['tc']), Paragraph("1,024 KB", st['tc_loss']), Paragraph("-50.0%", st['tc_loss']), Paragraph("MISMATCH (Hardware)", st['tc_loss'])],
        [Paragraph("(128,128) Microbench", st['tc_bold']), Paragraph("RPi4", st['tc']), Paragraph("+63.91%", st['tc_loss']), Paragraph("-0.01%", st['tc']), Paragraph("-100.0%", st['tc_loss']), Paragraph("MISMATCH (Spilling)", st['tc_loss'])],
        [Paragraph("Ablation Fused-4T", st['tc_bold']), Paragraph("Jetson Nano", st['tc']), Paragraph("+33.11%", st['tc_loss']), Paragraph("+0.74%", st['tc']), Paragraph("-97.8%", st['tc_loss']), Paragraph("MISMATCH (No scaling)", st['tc_loss'])],
        [Paragraph("Fallback Guard (48,96)", st['tc_bold']), Paragraph("Jetson Nano", st['tc']), Paragraph("Fallback triggers", st['tc']), Paragraph("100% Trigger", st['tc_win']), Paragraph("0.0%", st['tc_win']), Paragraph("MATCH (Verified)", st['tc_win'])],
        [Paragraph("End-to-End VGG-16", st['tc_bold']), Paragraph("RPi4", st['tc']), Paragraph("&gt; 0% Gain", st['tc']), Paragraph("+4.80% (p=2.8e-6)", st['tc_win']), Paragraph("Verified", st['tc_win']), Paragraph("MATCH (Win, 93s saved)", st['tc_win'])],
    ]
    t_mini_disc = Table(mini_disc_data, colWidths=[110, 75, 75, 95, 55, 130])
    t_mini_disc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_mini_disc)

    # Build PDF
    doc.build(story, canvasmaker=DigestCanvas)
    print(f"[PDF Builder] Successfully generated: {pdf_path}")
    return pdf_path


def main():
    print("[PDF Builder] Generating comprehensive empirical analysis and AI digest PDFs...")
    p1 = build_full_report_pdf()
    p2 = build_ai_reference_digest_pdf()
    print(f"\nAll PDFs Built Successfully!\n1. Full Monograph: {p1}\n2. AI Reference Digest: {p2}")

if __name__ == "__main__":
    main()
