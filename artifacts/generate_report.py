import os
import json
import csv
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def build_pdf():
    pdf_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/combined_experimental_results.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom Styles for Premium Look
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#2C5282'),
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=6
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#2D3748')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    # 1. Header/Title Block
    story.append(Paragraph("Cache-Aware Winograd Implementation", title_style))
    story.append(Paragraph("Combined Experimental Results and Platform Metadata Report", ParagraphStyle('Sub', parent=body_style, fontSize=11, textColor=colors.HexColor('#4A5568'))))
    story.append(Spacer(1, 15))
    
    # 2. Section 1: Platform & System Details
    story.append(Paragraph("1. Hardware & Software Platform Profile", h1_style))
    platform_json_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/platform_descriptor.json"
    if os.path.exists(platform_json_path):
        with open(platform_json_path, 'r') as f:
            plat = json.load(f)
        platform_data = [
            [Paragraph("Property", table_header_style), Paragraph("Value", table_header_style)],
            [Paragraph("OS / Architecture", table_cell_style), Paragraph(f"{plat.get('os')} / {plat.get('architecture')}", table_cell_style)],
            [Paragraph("CPU Model", table_cell_style), Paragraph(plat.get('cpu_model', 'N/A'), table_cell_style)],
            [Paragraph("Python Version", table_cell_style), Paragraph(plat.get('python_version', 'N/A'), table_cell_style)],
            [Paragraph("Logical / Physical Cores", table_cell_style), Paragraph(f"{plat.get('logical_cores')} / {plat.get('physical_cores')}", table_cell_style)],
            [Paragraph("L1 Data Cache Size", table_cell_style), Paragraph(f"{plat.get('l1d_size_bytes', 0) // 1024} KB ({plat.get('l1d_size_bytes')} B)", table_cell_style)],
            [Paragraph("L2 Cache Size", table_cell_style), Paragraph(f"{plat.get('l2_size_bytes', 0) // 1024} KB ({plat.get('l2_size_bytes')} B)", table_cell_style)],
            [Paragraph("Cache Line Size", table_cell_style), Paragraph(f"{plat.get('line_size_bytes')} B", table_cell_style)],
            [Paragraph("Git Commit Reference", table_cell_style), Paragraph(plat.get('git_commit', 'N/A')[:8], table_cell_style)],
            [Paragraph("Execution Timestamp", table_cell_style), Paragraph(plat.get('timestamp', 'N/A'), table_cell_style)]
        ]
        t_plat = Table(platform_data, colWidths=[200, 330])
        t_plat.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F7FAFC'), colors.HexColor('#EDF2F7')]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ]))
        story.append(t_plat)
    else:
        story.append(Paragraph("Platform descriptor file missing.", body_style))
        
    story.append(Spacer(1, 10))
    
    # 3. Section 2: End-to-End Run Status
    story.append(Paragraph("2. End-to-End Layer Run Status", h1_style))
    e2e_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/raw/e2e_runs_error.csv"
    if os.path.exists(e2e_path):
        e2e_rows = []
        with open(e2e_path, 'r') as f:
            reader = csv.reader(f)
            e2e_rows = list(reader)
        if len(e2e_rows) > 0:
            e2e_table_data = []
            # Header
            e2e_table_data.append([Paragraph(cell, table_header_style) for cell in e2e_rows[0]])
            # Body
            for row in e2e_rows[1:]:
                e2e_table_data.append([Paragraph(cell, table_cell_style) for cell in row])
            t_e2e = Table(e2e_table_data, colWidths=[150, 150, 230])
            t_e2e.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F7FAFC'), colors.HexColor('#EDF2F7')]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
            ]))
            story.append(t_e2e)
            story.append(Spacer(1, 5))
            story.append(Paragraph("<b>Note:</b> Full end-to-end framework benchmarks were reported as unsupported due to missing system-level dependencies for PyTorch/ONNX Runtime complete models on this specific environment, shifting the focus to high-fidelity microbenchmarks.", ParagraphStyle('NoteStyle', parent=body_style, fontSize=8, textColor=colors.HexColor('#718096'))))
        else:
            story.append(Paragraph("E2E runs error file is empty.", body_style))
    else:
        story.append(Paragraph("E2E runs error file not found.", body_style))
        
    story.append(Spacer(1, 15))
    
    # 4. Section 3: Autotiler Decisions & Parameters
    story.append(Paragraph("3. Autotiler Decision Configurations", h1_style))
    autotile_csv_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/processed/paper_table_autotiling.csv"
    if os.path.exists(autotile_csv_path):
        at_rows = []
        with open(autotile_csv_path, 'r') as f:
            reader = csv.reader(f)
            at_rows = list(reader)
        if len(at_rows) > 0:
            at_table_data = []
            # Select subset of columns to fit page nicely
            # Header: Workload, Target_Cache, Selected_Tile, Estimated_Working_Set_Bytes, Reuse_Score
            headers = ["Workload", "Target Cache", "Selected Tile", "Working Set (B)", "Reuse Score"]
            at_table_data.append([Paragraph(h, table_header_style) for h in headers])
            
            # Map index of columns we want
            # Workload (idx 4), Target_Cache (idx 9), Selected_Tile (idx 6), Estimated_Working_Set_Bytes (idx 10), Reuse_Score (idx 11)
            # Find headers indices
            orig_headers = at_rows[0]
            w_idx = orig_headers.index("Workload")
            tc_idx = orig_headers.index("Target_Cache")
            st_idx = orig_headers.index("Selected_Tile")
            ws_idx = orig_headers.index("Estimated_Working_Set_Bytes")
            rs_idx = orig_headers.index("Reuse_Score")
            
            seen_configs = set()
            for r in at_rows[1:]:
                # unique by (workload, target_cache)
                config_key = (r[w_idx], r[tc_idx])
                if config_key not in seen_configs:
                    seen_configs.add(config_key)
                    at_table_data.append([
                        Paragraph(r[w_idx], table_cell_style),
                        Paragraph(f"{int(r[tc_idx]) // 1024} KB", table_cell_style),
                        Paragraph(r[st_idx], table_cell_style),
                        Paragraph(f"{int(r[ws_idx]):,}", table_cell_style),
                        Paragraph(r[rs_idx][:6], table_cell_style),
                    ])
            t_at = Table(at_table_data, colWidths=[150, 90, 90, 100, 100])
            t_at.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F7FAFC'), colors.HexColor('#EDF2F7')]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
            ]))
            story.append(t_at)
        else:
            story.append(Paragraph("Autotiling decisions table is empty.", body_style))
    else:
        story.append(Paragraph("Autotiling decisions table not found.", body_style))
        
    story.append(PageBreak())
    
    # 5. Section 4: Microbenchmark Performance Comparisons
    story.append(Paragraph("4. Microbenchmark Detailed Perf & Statistical Validation", h1_style))
    story.append(Paragraph("Measurements compiled over 30 runs per configuration comparing the Cache-Aware Fused Winograd kernel to the non-fused baseline implementation.", body_style))
    
    microbench_csv_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/processed/paper_table_microbench.csv"
    if os.path.exists(microbench_csv_path):
        mb_rows = []
        with open(microbench_csv_path, 'r') as f:
            reader = csv.reader(f)
            mb_rows = list(reader)
        if len(mb_rows) > 0:
            mb_table_data = []
            # Select subset of columns: C_in, C_out, Fused, MultiCore, Mean_Latency_ms, Improvement_vs_Baseline_pct, P_Value_vs_Baseline
            headers = ["C_in", "C_out", "Fused", "Threads", "Mean Lat (ms)", "Improv %", "p-value"]
            mb_table_data.append([Paragraph(h, table_header_style) for h in headers])
            
            orig_headers = mb_rows[0]
            cin_idx = orig_headers.index("C_in")
            cout_idx = orig_headers.index("C_out")
            fused_idx = orig_headers.index("Fused")
            mc_idx = orig_headers.index("MultiCore")
            mean_idx = orig_headers.index("Mean_Latency_ms")
            imp_idx = orig_headers.index("Improvement_vs_Baseline_pct")
            p_idx = orig_headers.index("P_Value_vs_Baseline")
            
            for r in mb_rows[1:]:
                # Thread representation
                threads = "4" if r[mc_idx] == "True" else "1"
                # Display values nicely
                imp_val = r[imp_idx]
                if imp_val == "N/A":
                    imp_disp = "-"
                else:
                    val = float(imp_val)
                    sign = "+" if val >= 0 else ""
                    imp_disp = f"{sign}{val:.2f}%"
                    
                mb_table_data.append([
                    Paragraph(r[cin_idx], table_cell_style),
                    Paragraph(r[cout_idx], table_cell_style),
                    Paragraph("Yes" if r[fused_idx] == "True" else "No", table_cell_style),
                    Paragraph(threads, table_cell_style),
                    Paragraph(f"{float(r[mean_idx]):.4f}", table_cell_style),
                    Paragraph(imp_disp, table_cell_style),
                    Paragraph(r[p_idx], table_cell_style),
                ])
            t_mb = Table(mb_table_data, colWidths=[60, 60, 60, 60, 100, 100, 130])
            t_mb.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F7FAFC'), colors.HexColor('#EDF2F7')]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
            ]))
            story.append(t_mb)
        else:
            story.append(Paragraph("Microbenchmark details table is empty.", body_style))
    else:
        story.append(Paragraph("Microbenchmark details table not found.", body_style))
        
    story.append(Spacer(1, 15))
    
    # 6. Section 5: Backend Comparison vs ONNX Runtime
    story.append(Paragraph("5. System-Level Backend Comparison", h1_style))
    comp_csv_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/comparisons/backend_comparison_combined.csv"
    if os.path.exists(comp_csv_path):
        comp_rows = []
        with open(comp_csv_path, 'r') as f:
            reader = csv.reader(f)
            comp_rows = list(reader)
        if len(comp_rows) > 0:
            comp_table_data = []
            headers = ["Backend", "Regime (C_in x C_out)", "Avg Latency (ms)", "Std Dev (ms)", "Speedup / Note"]
            comp_table_data.append([Paragraph(h, table_header_style) for h in headers])
            
            orig_headers = comp_rows[0]
            backend_idx = orig_headers.index("backend")
            avg_idx = orig_headers.index("avg_latency_ms")
            std_idx = orig_headers.index("std_latency_ms")
            cin_idx = orig_headers.index("c_in")
            cout_idx = orig_headers.index("c_out")
            note_idx = orig_headers.index("notes")
            
            for r in comp_rows[1:]:
                regime = f"{r[cin_idx]} \u2192 {r[cout_idx]}"
                comp_table_data.append([
                    Paragraph(r[backend_idx], table_cell_style),
                    Paragraph(regime, table_cell_style),
                    Paragraph(f"{float(r[avg_idx]):.4f}", table_cell_style),
                    Paragraph(f"{float(r[std_idx]):.4f}", table_cell_style),
                    Paragraph(r[note_idx], table_cell_style)
                ])
            t_comp = Table(comp_table_data, colWidths=[80, 110, 100, 90, 150])
            t_comp.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F7FAFC'), colors.HexColor('#EDF2F7')]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
            ]))
            story.append(t_comp)
        else:
            story.append(Paragraph("Backend comparison table is empty.", body_style))
    else:
        story.append(Paragraph("Backend comparison table not found.", body_style))
        
    story.append(PageBreak())
    
    # 7. Section 6: Experimental Plots
    story.append(Paragraph("6. Performance Visualization Plots", h1_style))
    plot_dir = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/plots"
    plots = [
        ("paper_fig_microbench_improvement.png", "Figure 1: Percentage Latency Improvement/Regression relative to non-fused baseline across different workload shapes"),
        ("paper_fig_microbench_ci.png", "Figure 2: Confidence intervals (95%) and variance distribution across baseline and cache-aware fused runs"),
        ("paper_fig_microbench_latency.png", "Figure 3: Absolute Latency (ms) comparison for Baseline vs Fused Winograd kernels"),
        ("paper_fig_autotiling_decisions.png", "Figure 4: Autotiler working set sizes and reuse score estimates vs L1/L2 Cache Capacities")
    ]
    
    for filename, caption in plots:
        plot_path = os.path.join(plot_dir, filename)
        if os.path.exists(plot_path):
            # Keep each plot and its caption together
            img = Image(plot_path, width=5.5*inch, height=3*inch)
            cap = Paragraph(f"<i>{caption}</i>", ParagraphStyle('PlotCap', parent=body_style, fontSize=8, alignment=1, textColor=colors.HexColor('#4A5568')))
            story.append(KeepTogether([img, Spacer(1, 4), cap, Spacer(1, 15)]))
        else:
            story.append(Paragraph(f"Plot missing: {filename}", body_style))
            
    story.append(PageBreak())
    
    # 8. Appendix: Combined Raw Latencies (840 Runs)
    story.append(Paragraph("Appendix: Combined Raw Latency Run Logs", h1_style))
    story.append(Paragraph("Listing of all raw latencies (ms) recorded across individual execution trials. The full dataset is printed below for complete auditability.", body_style))
    
    raw_csv_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/raw/microbenchmark_raw_latencies.csv"
    if os.path.exists(raw_csv_path):
        raw_rows = []
        with open(raw_csv_path, 'r') as f:
            reader = csv.reader(f)
            raw_rows = list(reader)
        
        if len(raw_rows) > 1:
            # Format raw logs into a compact 4-column layout to save space
            # columns: Config (C_in->C_out, Fused, Thread), RunID, Latency (ms)
            # We pack 3 tuples per row in the report table
            raw_data_items = []
            
            orig_headers = raw_rows[0]
            cin_idx = orig_headers.index("c_in")
            cout_idx = orig_headers.index("c_out")
            fused_idx = orig_headers.index("fused")
            th_idx = orig_headers.index("threads")
            rid_idx = orig_headers.index("run_id")
            lat_idx = orig_headers.index("latency_ms")
            
            for r in raw_rows[1:]:
                if not r: continue
                config_str = f"{r[cin_idx]}\u2192{r[cout_idx]} (F={r[fused_idx][0]}, T={r[th_idx]})"
                run_id = r[rid_idx]
                lat_ms = f"{float(r[lat_idx]):.4f}"
                raw_data_items.append((config_str, run_id, lat_ms))
            
            # Pack into a 3-column group table
            col_headers = ["Config", "Run", "Latency", "Config", "Run", "Latency", "Config", "Run", "Latency"]
            grouped_table_data = [[Paragraph(h, table_header_style) for h in col_headers]]
            
            num_items = len(raw_data_items)
            # split into 3 columns
            col_len = (num_items + 2) // 3
            
            for i in range(col_len):
                row_cells = []
                for col_idx in range(3):
                    item_idx = i + col_idx * col_len
                    if item_idx < num_items:
                        item = raw_data_items[item_idx]
                        row_cells.extend([
                            Paragraph(item[0], table_cell_style),
                            Paragraph(item[1], table_cell_style),
                            Paragraph(item[2], table_cell_style)
                        ])
                    else:
                        row_cells.extend([Paragraph("", table_cell_style), Paragraph("", table_cell_style), Paragraph("", table_cell_style)])
                grouped_table_data.append(row_cells)
            
            t_raw = Table(grouped_table_data, colWidths=[100, 30, 45, 100, 30, 45, 100, 30, 45])
            t_raw.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4A5568')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                ('TOPPADDING', (0,0), (-1,-1), 2),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F7FAFC')]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ]))
            story.append(t_raw)
        else:
            story.append(Paragraph("Raw latencies file has no run rows.", body_style))
    else:
        story.append(Paragraph("Raw latencies file not found.", body_style))
        
    doc.build(story)
    print("PDF generation completed successfully.")

def build_markdown():
    md_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/combined_results_for_claude.md"
    content = []
    
    content.append("# Cache-Aware Winograd Implementation: Combined Experimental Results")
    content.append("\nThis document contains all experimental results, statistical analysis, and system metadata in a clean Markdown format optimized for LLMs (like Claude) to parse.")
    content.append("\n---\n")
    
    # 1. Platform & System Details
    content.append("## 1. Hardware & Platform Profile")
    platform_json_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/platform_descriptor.json"
    if os.path.exists(platform_json_path):
        with open(platform_json_path, 'r') as f:
            plat = json.load(f)
        content.append(f"- **OS:** {plat.get('os')}")
        content.append(f"- **Architecture:** {plat.get('architecture')}")
        content.append(f"- **CPU Model:** {plat.get('cpu_model')}")
        content.append(f"- **Python Version:** {plat.get('python_version')}")
        content.append(f"- **Logical / Physical Cores:** {plat.get('logical_cores')} / {plat.get('physical_cores')}")
        content.append(f"- **L1 Data Cache Size:** {plat.get('l1d_size_bytes')} bytes ({plat.get('l1d_size_bytes', 0) // 1024} KB)")
        content.append(f"- **L2 Cache Size:** {plat.get('l2_size_bytes')} bytes ({plat.get('l2_size_bytes', 0) // 1024} KB)")
        content.append(f"- **Cache Line Size:** {plat.get('line_size_bytes')} bytes")
        content.append(f"- **Git Commit:** `{plat.get('git_commit')}`")
        content.append(f"- **Timestamp:** {plat.get('timestamp')}")
    else:
        content.append("Platform details file missing.")
        
    content.append("\n---\n")
    
    # 2. End-to-End Run Status
    content.append("## 2. End-to-End Layer Run Status")
    e2e_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/raw/e2e_runs_error.csv"
    if os.path.exists(e2e_path):
        with open(e2e_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if rows:
            content.append("| " + " | ".join(rows[0]) + " |")
            content.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
            for row in rows[1:]:
                content.append("| " + " | ".join(row) + " |")
        else:
            content.append("E2E runs file is empty.")
    else:
        content.append("E2E runs error file not found.")
        
    content.append("\n---\n")
    
    # 3. Autotiler Decisions & Parameters
    content.append("## 3. Autotiler Decisions & Model Configuration")
    autotile_csv_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/processed/paper_table_autotiling.csv"
    if os.path.exists(autotile_csv_path):
        with open(autotile_csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if rows:
            headers = ["Workload", "Target Cache", "Selected Tile", "Estimated Working Set (B)", "Reuse Score"]
            content.append("| " + " | ".join(headers) + " |")
            content.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            orig_headers = rows[0]
            w_idx = orig_headers.index("Workload")
            tc_idx = orig_headers.index("Target_Cache")
            st_idx = orig_headers.index("Selected_Tile")
            ws_idx = orig_headers.index("Estimated_Working_Set_Bytes")
            rs_idx = orig_headers.index("Reuse_Score")
            
            seen_configs = set()
            for r in rows[1:]:
                config_key = (r[w_idx], r[tc_idx])
                if config_key not in seen_configs:
                    seen_configs.add(config_key)
                    content.append(f"| {r[w_idx]} | {int(r[tc_idx]) // 1024} KB | {r[st_idx]} | {int(r[ws_idx]):,} | {r[rs_idx][:6]} |")
        else:
            content.append("Autotiler table is empty.")
    else:
        content.append("Autotiler CSV file not found.")
        
    content.append("\n---\n")
    
    # 4. Microbenchmark Performance Comparisons
    content.append("## 4. Microbenchmark Performance & Statistical Verification")
    microbench_csv_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/processed/paper_table_microbench.csv"
    if os.path.exists(microbench_csv_path):
        with open(microbench_csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if rows:
            headers = ["C_in", "C_out", "Fused", "Threads", "Mean Latency (ms)", "Improvement %", "p-value"]
            content.append("| " + " | ".join(headers) + " |")
            content.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            orig_headers = rows[0]
            cin_idx = orig_headers.index("C_in")
            cout_idx = orig_headers.index("C_out")
            fused_idx = orig_headers.index("Fused")
            mc_idx = orig_headers.index("MultiCore")
            mean_idx = orig_headers.index("Mean_Latency_ms")
            imp_idx = orig_headers.index("Improvement_vs_Baseline_pct")
            p_idx = orig_headers.index("P_Value_vs_Baseline")
            
            for r in rows[1:]:
                threads = "4" if r[mc_idx] == "True" else "1"
                imp_val = r[imp_idx]
                if imp_val == "N/A":
                    imp_disp = "-"
                else:
                    val = float(imp_val)
                    sign = "+" if val >= 0 else ""
                    imp_disp = f"{sign}{val:.2f}%"
                content.append(f"| {r[cin_idx]} | {r[cout_idx]} | {'Yes' if r[fused_idx] == 'True' else 'No'} | {threads} | {float(r[mean_idx]):.4f} | {imp_disp} | {r[p_idx]} |")
        else:
            content.append("Microbenchmark table is empty.")
    else:
        content.append("Microbenchmark CSV file not found.")
        
    content.append("\n---\n")
    
    # 5. System-Level Backend Comparison
    content.append("## 5. System-Level Backend Comparison (vs ONNX Runtime)")
    comp_csv_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/comparisons/backend_comparison_combined.csv"
    if os.path.exists(comp_csv_path):
        with open(comp_csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if rows:
            headers = ["Backend", "Regime (C_in \u2192 C_out)", "Avg Latency (ms)", "Std Dev (ms)", "Notes"]
            content.append("| " + " | ".join(headers) + " |")
            content.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            orig_headers = rows[0]
            backend_idx = orig_headers.index("backend")
            avg_idx = orig_headers.index("avg_latency_ms")
            std_idx = orig_headers.index("std_latency_ms")
            cin_idx = orig_headers.index("c_in")
            cout_idx = orig_headers.index("c_out")
            note_idx = orig_headers.index("notes")
            
            for r in rows[1:]:
                regime = f"{r[cin_idx]} \u2192 {r[cout_idx]}"
                content.append(f"| {r[backend_idx]} | {regime} | {float(r[avg_idx]):.4f} | {float(r[std_idx]):.4f} | {r[note_idx]} |")
        else:
            content.append("Backend comparison table is empty.")
    else:
        content.append("Backend comparison CSV file not found.")
        
    content.append("\n---\n")
    
    # 6. Aggregated Statistics / Summary of Raw Runs
    content.append("## 6. Summary Statistics of Raw Trials")
    content.append("Below are the aggregated statistics across all 840 trials grouped by configuration:")
    raw_csv_path = "/Users/ishaanupponi/Documents/My projects/Cache-Aware-Wino-Implementation/artifacts/raw/microbenchmark_raw_latencies.csv"
    if os.path.exists(raw_csv_path):
        stats = {}
        with open(raw_csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            cin_idx = header.index("c_in")
            cout_idx = header.index("c_out")
            fused_idx = header.index("fused")
            th_idx = header.index("threads")
            lat_idx = header.index("latency_ms")
            
            for row in reader:
                if not row: continue
                cfg = f"C_in={row[cin_idx]}, C_out={row[cout_idx]}, Fused={row[fused_idx]}, Threads={row[th_idx]}"
                lat = float(row[lat_idx])
                if cfg not in stats:
                    stats[cfg] = []
                stats[cfg].append(lat)
                
        headers = ["Configuration", "Count", "Mean (ms)", "Min (ms)", "Max (ms)", "Std Dev (ms)"]
        content.append("| " + " | ".join(headers) + " |")
        content.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for cfg, lats in sorted(stats.items()):
            count = len(lats)
            mean_lat = sum(lats) / count
            min_lat = min(lats)
            max_lat = max(lats)
            var_lat = sum((x - mean_lat) ** 2 for x in lats) / max(1, count - 1)
            std_lat = var_lat ** 0.5
            content.append(f"| {cfg} | {count} | {mean_lat:.5f} | {min_lat:.5f} | {max_lat:.5f} | {std_lat:.5f} |")
    else:
        content.append("Raw latency log not found.")
        
    with open(md_path, 'w') as f:
        f.write("\n".join(content))
    print("Markdown generation completed successfully.")

if __name__ == '__main__':
    build_pdf()
    build_markdown()

