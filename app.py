"""SCHOTT Engineering Drawing Intelligence — Streamlit UI (Enterprise MVP).

Run with: streamlit run app.py
"""
from __future__ import annotations

import sys
import json
import io
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.pipeline import process_email, process_pdf, render_pdf_page_to_image

# ================================================================
# Page configuration
# ================================================================
st.set_page_config(
    page_title="SCHOTT Drawing Intelligence",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# Header
# ================================================================
st.title("📐 SCHOTT Engineering Drawing Intelligence")
st.caption(
    "AI-assisted extraction of engineering drawings into validated, SAP-ready material data"
)

# ================================================================
# Sidebar — grouped scenarios
# ================================================================
st.sidebar.header("Input source")

mode = st.sidebar.radio(
    "Select input type",
    ["📧 Incoming email (.eml)", "📄 Standalone CAD drawing"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

DEMO_EMAIL_GROUPS = {
    "🟢 Normal processing": [
        ("Happy path — clean order", "01_happy_path.eml"),
    ],
    "🔒 Security": [
        ("Suspicious instructions in email", "02_injection_in_email.eml"),
        ("Suspicious instructions in CAD drawing", "03_injection_in_pdf.eml"),
    ],
    "📄 Document quality": [
        ("Corrupt drawing file", "04_corrupt_pdf.eml"),
        ("Missing attachment", "05_no_attachment.eml"),
    ],
    "⚖️ Business validation": [
        ("Email vs. CAD drawing conflict", "06_cross_source_conflict.eml"),
        ("Incomplete drawing", "07_low_quality.eml"),
        ("Poor image quality", "08_blurred_drawing.eml"),
    ],
}

DEMO_FLAT = [
    (f"{group}  •  {label}", filename)
    for group, items in DEMO_EMAIL_GROUPS.items()
    for label, filename in items
]

DEMO_FOLDER = Path("data/incoming_emails")

source_path: Path | None = None
run_button = False

if mode == "📧 Incoming email (.eml)":
    st.sidebar.subheader("Choose demo scenario")
    selected_label = st.sidebar.selectbox(
        "Demo scenario",
        [label for label, _ in DEMO_FLAT],
        label_visibility="collapsed",
    )
    selected_file = dict(DEMO_FLAT)[selected_label]

    st.sidebar.markdown("_or upload your own:_")
    uploaded_eml = st.sidebar.file_uploader("Upload .eml", type=["eml"], key="eml_up")

    run_button = st.sidebar.button("▶️ Process email", type="primary", use_container_width=True)

    if run_button:
        if uploaded_eml:
            temp_path = DEMO_FOLDER / f"_uploaded_{uploaded_eml.name}"
            temp_path.write_bytes(uploaded_eml.read())
            source_path = temp_path
        else:
            source_path = DEMO_FOLDER / selected_file

else:  # PDF mode
    st.sidebar.subheader("Upload a CAD drawing")
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="pdf_up",
        help="Test the extraction workflow against a standalone drawing.",
    )
    run_button = st.sidebar.button("▶️ Process drawing", type="primary", use_container_width=True)

    if run_button and uploaded_pdf:
        temp_dir = Path("data/uploaded_pdfs")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"_uploaded_{uploaded_pdf.name}"
        temp_path.write_bytes(uploaded_pdf.read())
        source_path = temp_path

# ================================================================
# Landing state
# ================================================================
if not run_button or source_path is None:
    st.info("👈 Select an input in the sidebar and click Process to begin.")

    st.markdown("### What this system does")
    st.markdown("""
    - 🧠 **Uses Vision AI** to understand customer CAD drawings and extract engineering specifications
    - ✅ **Validates extracted information** using deterministic engineering rules
    - 🛡️ **Detects inconsistencies and security risks** before they reach ERP
    - 👥 **Routes uncertain cases** for human review
    - 📤 **Generates SAP-ready output** — only when safe to do so
    """)

    st.markdown("---")
    st.markdown(
        """
        <div style='background-color: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 16px;'>
            <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                <span style='font-size: 20px; margin-right: 12px;'>✨</span>
                <span style='font-size: 15px; color: #111827;'><strong>AI</strong> understands the drawing.</span>
            </div>
            <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                <span style='font-size: 20px; margin-right: 12px;'>🛡️</span>
                <span style='font-size: 15px; color: #111827;'><strong>Engineering rules</strong> verify the result.</span>
            </div>
            <div style='display: flex; align-items: center;'>
                <span style='font-size: 20px; margin-right: 12px;'>🗄️</span>
                <span style='font-size: 15px; color: #111827;'>Only <strong>trusted data</strong> reaches SAP.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔧 Technical details"):
        st.markdown("""
        - **Prompt Injection Defence** — deterministic detector runs on all text sources before any AI call
        - **Vision AI Extraction** — Gemini 2.5 Flash reads the CAD drawing
        - **Engineering Rules Engine** — mandatory fields, dimensional ranges, geometric plausibility
        - **Cross-source Reconciliation** — compares customer email against extracted drawing data
        - **Human Review Workflow** — every ambiguous case escalated with an actionable reason
        - **SAP Integration** — structured output generated only when all validation layers pass
        """)
    st.stop()

# ================================================================
# Run the workflow
# ================================================================
with st.spinner(f"Processing {source_path.name}..."):
    if mode == "📧 Incoming email (.eml)":
        result = process_email(source_path)
    else:
        result = process_pdf(source_path)

# ================================================================
# DECISION BANNER
# ================================================================
status = result.pipeline_status

if status == "clean":
    banner_color = "#166534"
    banner_bg = "#dcfce7"
    banner_icon = "✅"
    banner_title = "READY FOR SAP"
    banner_subtitle = "The engineering drawing passed all validation checks and is ready for SAP."
elif status == "review_required":
    banner_color = "#854d0e"
    banner_bg = "#fef3c7"
    banner_icon = "⚠️"
    banner_title = "HUMAN REVIEW REQUIRED"
    banner_subtitle = "One or more validation checks require human review before SAP processing."
else:  # rejected
    banner_color = "#991b1b"
    banner_bg = "#fee2e2"
    banner_icon = "❌"
    banner_title = "PROCESSING FAILED"
    banner_subtitle = "The document could not be processed successfully. Customer action is required."

st.markdown(
    f"""
    <div style='
        background-color: {banner_bg};
        border-left: 8px solid {banner_color};
        padding: 20px 24px;
        border-radius: 6px;
        margin-bottom: 20px;
    '>
        <div style='font-size: 28px; font-weight: 700; color: {banner_color}; margin-bottom: 4px;'>
            {banner_icon} {banner_title}
        </div>
        <div style='font-size: 15px; color: {banner_color}; opacity: 0.85;'>
            {banner_subtitle}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ================================================================
# Three-column layout
# ================================================================
col_email, col_pdf, col_decision = st.columns(3, gap="medium")

# ---------- Column 1: Email ----------
with col_email:
    st.subheader("📧 Customer email")
    if result.parsed_email:
        pe = result.parsed_email
        st.markdown(f"**From:** {pe.from_addr or '(none)'}")
        st.markdown(f"**Subject:** {pe.subject or '(none)'}")
        if pe.date:
            st.caption(f"Date: {pe.date}")

        st.markdown("**Body:**")
        body_preview = pe.body_text[:700]
        if len(pe.body_text) > 700:
            body_preview += "..."
        st.text_area("body", body_preview, height=180, label_visibility="collapsed", disabled=True)

        if pe.attachment_paths:
            st.markdown("**Attachments:**")
            for a in pe.attachment_paths:
                st.code(Path(a).name, language=None)
        else:
            st.error("❌ No attachments found in this email")

        # Email security alert — plain language
        if result.injection_scan_body and result.injection_scan_body.get("injection_detected"):
            st.warning(
                "⚠️ **Security validation triggered**\n\n"
                "The customer email contains text attempting to influence the AI extraction. "
                "The suspicious content was ignored automatically and the message has been routed for human review."
            )
            with st.expander("🔧 Technical details"):
                st.caption(f"Severity: **{result.injection_scan_body['severity']}**")
                st.caption(f"Matches found: {len(result.injection_scan_body['matches'])}")
                for m in result.injection_scan_body["matches"]:
                    st.code(f"[{m['category']}] {m['matched_text']}")
    else:
        st.info("Standalone drawing mode — no email source")

# ---------- Column 2: CAD Drawing ----------
with col_pdf:
    st.subheader("📄 CAD drawing")

    # Drawing quality — verbal label only
    if result.image_quality_score is not None:
        q = result.image_quality_score
        if q >= 1.0:
            st.success("✓ **Drawing quality:** Good")
        elif q >= 0.5:
            st.warning("⚠️ **Drawing quality:** Acceptable")
        else:
            st.error("✗ **Drawing quality:** Poor — extraction not reliable")

    # Try to render first-page preview of the PDF
    if result.pdf_parseable and result.parsed_email and result.parsed_email.first_pdf_attachment:
        try:
            pdf_path = Path(result.parsed_email.first_pdf_attachment)
            preview_img = render_pdf_page_to_image(pdf_path, dpi=100)
            st.image(preview_img, caption=pdf_path.name, use_container_width=True)
        except Exception:
            pass
    elif not result.parsed_email and result.pdf_parseable:
        # PDF-only mode — render from the direct source_path
        try:
            preview_img = render_pdf_page_to_image(Path(source_path), dpi=100)
            st.image(preview_img, caption=Path(source_path).name, use_container_width=True)
        except Exception:
            pass

    # PDF security alert — plain language with reassurance
    if result.injection_scan_pdf and result.injection_scan_pdf.get("injection_detected"):
        st.warning(
            "⚠️ **Security validation triggered**\n\n"
            "Suspicious instructions were found inside the CAD drawing. "
            "These instructions were ignored automatically before AI extraction. "
            "Engineering dimensions and drawing information were still extracted successfully — "
            "only the suspicious instruction text was ignored."
        )
        with st.expander("🔧 Technical details"):
            st.caption(f"Severity: **{result.injection_scan_pdf['severity']}**")
            st.caption(f"Matches found: {len(result.injection_scan_pdf['matches'])}")
            for m in result.injection_scan_pdf["matches"]:
                st.code(f"[{m['category']}] {m['matched_text']}")

    # Extracted fields
    if result.pdf_extraction:
        st.markdown("**Extracted specifications:**")
        ext = result.pdf_extraction
        st.markdown(f"- **Drawing number:** `{ext.get('drawing_number')}`")
        st.markdown(f"- **Material:** `{ext.get('material')}`")
        st.markdown(f"- **Overall length:** `{ext.get('overall_length_mm')} mm`")
        st.markdown(f"- **Overall width:** `{ext.get('overall_width_mm')} mm`")
        st.markdown(f"- **Drawing date:** `{ext.get('date_iso')}`")
        st.markdown(f"- **Created by:** `{ext.get('created_by')}`")

        zones = ext.get("cooking_zones", [])
        st.markdown(f"**Cooking zones detected: {len(zones)}/4**")
        for z in zones:
            st.markdown(f"  - {z.get('zone_id')}: Ø {z.get('diameter_mm')} mm")

        with st.expander("🔧 Full extraction details"):
            st.json(ext)
    elif result.pdf_parseable is False and result.pdf_present:
        st.error("The drawing file could not be opened.")
    elif not result.pdf_present:
        st.info("No CAD drawing attachment in this email.")
    elif result.image_quality_score is not None and result.image_quality_score < 0.5:
        st.warning("AI extraction was skipped due to low drawing quality.")

# ---------- Column 3: Decision Support ----------
# ---------- Column 3: Decision Support ----------
with col_decision:
    st.subheader("🔍 Decision Support")

    # Overall Confidence — label follows pipeline decision, not raw numeric buckets
    if result.pdf_extraction and result.extraction_quality_score is not None:
        llm_conf = result.pdf_extraction.get("overall_confidence", 0.0)
        det_score = result.extraction_quality_score
        overall_confidence = min(llm_conf, det_score)

        if status == "clean":
            conf_label = "High"
            conf_color = "#166534"
            conf_sentence = "The engineering drawing was successfully extracted and validated."
        elif status == "review_required" and overall_confidence >= 0.5:
            conf_label = "Medium"
            conf_color = "#854d0e"
            conf_sentence = ("The engineering drawing was extracted successfully. "
                             "Human review is required because one or more validation checks were triggered.")
        else:
            conf_label = "Low"
            conf_color = "#991b1b"
            conf_sentence = "The document could not be extracted with sufficient confidence."

        st.markdown(
            f"**Overall Confidence:** <span style='color: {conf_color}; font-weight: 700; font-size: 18px;'>{conf_label}</span>",
            unsafe_allow_html=True,
        )
        st.caption(conf_sentence)
    elif status == "rejected":
        st.markdown(
            "**Overall Confidence:** <span style='color: #991b1b; font-weight: 700; font-size: 18px;'>Low</span>",
            unsafe_allow_html=True,
        )
        st.progress(0.0)
        st.caption("The document could not be extracted with sufficient confidence.")

    st.markdown("---")

    # ---- Completed checks ----
    completed = []
    if result.pdf_present:
        completed.append("Attachment received")
    if result.pdf_parseable:
        completed.append("CAD drawing successfully processed")
    if result.image_quality_score is not None and result.image_quality_score >= 0.5:
        completed.append("Drawing quality acceptable")
    inj_body = result.injection_scan_body and result.injection_scan_body.get("injection_detected")
    inj_pdf = result.injection_scan_pdf and result.injection_scan_pdf.get("injection_detected")
    if (result.parsed_email or result.pdf_parseable) and not (inj_body or inj_pdf):
        completed.append("Security validation passed")
    if result.pdf_extraction and (result.extraction_quality_score or 0) >= 0.7:
        completed.append("Required engineering information extracted")
    if result.engineering_validation and result.engineering_validation.get("is_valid", True):
        error_count = sum(1 for i in result.engineering_validation.get("issues", [])
                          if i["severity"] == "error")
        if error_count == 0:
            completed.append("Engineering validation completed")
    if result.pdf_extraction and result.email_hints and not result.cross_source_issues:
        completed.append("Email and CAD drawing consistent")

    if completed:
        st.markdown("**✅ Completed checks**")
        for c in completed:
            st.markdown(
                f"<div style='color: #166534; font-size: 14px; margin: 3px 0;'>✓ {c}</div>",
                unsafe_allow_html=True,
            )

    # ---- Issues detected ----
    issues_display = []
    if not result.pdf_present:
        issues_display.append("No CAD drawing attached — customer follow-up required")
    if result.pdf_present and not result.pdf_parseable:
        issues_display.append("CAD drawing file could not be opened — request re-send")
    if result.image_quality_score is not None and result.image_quality_score < 0.5:
        issues_display.append("Drawing quality too low for reliable extraction")
    if inj_body:
        issues_display.append("Suspicious instructions detected in customer email")
    if inj_pdf:
        issues_display.append("Suspicious instructions detected inside CAD drawing")
    if result.extraction_quality_score is not None and result.extraction_quality_score < 0.7:
        issues_display.append("Some engineering information could not be extracted")
    if result.engineering_validation:
        for i in result.engineering_validation.get("issues", []):
            issues_display.append(f"Engineering rule flagged: {i['message']}")
    if result.cross_source_issues:
        issues_display.append(f"Email and CAD drawing disagree on {len(result.cross_source_issues)} field(s)")

    if issues_display:
        st.markdown("**⚠️ Issues detected**")
        for msg in issues_display:
            st.markdown(
                f"""
                <div style='
                    background-color: #fef3c7;
                    border-left: 3px solid #d97706;
                    padding: 8px 12px;
                    margin: 4px 0;
                    border-radius: 3px;
                    font-size: 14px;
                    color: #78350f;
                '>
                    ⚠ {msg}
                </div>
                """,
                unsafe_allow_html=True,
            )

# ================================================================
# Cross-source comparison cards
# ================================================================
if result.pdf_extraction and result.email_hints:
    st.markdown("---")
    st.subheader("🔁 Email vs. CAD drawing comparison")

    ext = result.pdf_extraction
    hints = result.email_hints

    # Build a unified comparison list
    fields_to_compare = [
        ("Material", hints.get("material"), ext.get("material")),
        ("Drawing number", hints.get("drawing_number"), ext.get("drawing_number")),
        ("Overall length (mm)", hints.get("overall_length_mm"), ext.get("overall_length_mm")),
        ("Overall width (mm)", hints.get("overall_width_mm"), ext.get("overall_width_mm")),
    ]

    # Also include zones if any
    email_zones = hints.get("cooking_zones") or []
    pdf_zones = ext.get("cooking_zones") or []
    email_zone_map = {z.get("zone_id"): z.get("diameter_mm") for z in email_zones if isinstance(z, dict)}
    pdf_zone_map = {z.get("zone_id"): z.get("diameter_mm") for z in pdf_zones if isinstance(z, dict)}
    for zid in ["top-left", "top-right", "bottom-left", "bottom-right"]:
        if zid in email_zone_map or zid in pdf_zone_map:
            fields_to_compare.append(
                (f"Zone {zid} (mm)", email_zone_map.get(zid), pdf_zone_map.get(zid))
            )

    # Which of these are in the conflict list?
    conflict_fields = set()
    for c in result.cross_source_issues:
        conflict_fields.add(c["field"])

    # Render cards in a grid
    cards_per_row = 2
    for i in range(0, len(fields_to_compare), cards_per_row):
        cols = st.columns(cards_per_row)
        for j, (field, email_val, pdf_val) in enumerate(fields_to_compare[i:i+cards_per_row]):
            with cols[j]:
                is_conflict = (
                    field.lower().replace(" ", "_").replace("(mm)", "").strip("_") in
                    {f.lower() for f in conflict_fields}
                ) or any(cf in field.lower() for cf in ["material", "drawing", "length", "width", "zone"] if cf in " ".join(conflict_fields).lower())

                # Simpler: check if any conflict message mentions this field
                relevant_conflict = None
                for c in result.cross_source_issues:
                    if field.lower().split("(")[0].strip().replace(" ", "_") in c["field"].lower():
                        relevant_conflict = c
                        break
                    if "zone" in field.lower() and "zone" in c["field"].lower():
                        zid = field.lower().split("zone ")[1].split(" ")[0] if "zone " in field.lower() else ""
                        if zid and zid in c["field"].lower():
                            relevant_conflict = c
                            break

                is_conflict = relevant_conflict is not None

                # Determine match status
                if email_val is None and pdf_val is None:
                    continue
                elif email_val is None:
                    status_line = "Only in CAD drawing"
                    status_color = "#6b7280"
                elif pdf_val is None:
                    status_line = "Only in email"
                    status_color = "#6b7280"
                elif is_conflict:
                    status_line = "⚠ Human review required before SAP processing"
                    status_color = "#854d0e"
                else:
                    status_line = "✓ Match"
                    status_color = "#166534"

                bg = "#fef3c7" if is_conflict else "#f9fafb"
                border = "#d97706" if is_conflict else "#e5e7eb"

                st.markdown(
                    f"""
                    <div style='
                        background-color: {bg};
                        border: 1px solid {border};
                        padding: 12px 16px;
                        border-radius: 6px;
                        margin-bottom: 8px;
                    '>
                        <div style='font-weight: 600; font-size: 15px; margin-bottom: 6px;'>{field}</div>
                        <div style='font-size: 13px; color: #374151;'>📧 Email: <code>{email_val if email_val is not None else "—"}</code></div>
                        <div style='font-size: 13px; color: #374151;'>📄 CAD Drawing: <code>{pdf_val if pdf_val is not None else "—"}</code></div>
                        <div style='font-size: 13px; color: {status_color}; margin-top: 6px; font-weight: 500;'>{status_line}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ================================================================
# SAP Output — single JSON, GREEN only
# ================================================================
st.markdown("---")
st.subheader("📤 SAP Output")

if result.pdf_extraction and status == "clean":
    ext = result.pdf_extraction

    sap_payload = {
        "customer_drawing_number": ext.get("drawing_number"),
        "material": ext.get("material"),
        "printing_color": ext.get("printing_color"),
        "overall_length_mm": ext.get("overall_length_mm"),
        "overall_width_mm": ext.get("overall_width_mm"),
        "cooking_zones": ext.get("cooking_zones", []),
        "line_thickness_notes": ext.get("line_thickness_notes"),
        "status": "READY_FOR_SAP",
    }

    st.success("✓ SAP JSON generated — ready for downstream ERP integration")
    st.json(sap_payload, expanded=True)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Download SAP JSON",
            data=json.dumps(sap_payload, indent=2),
            file_name=f"sap_{Path(source_path).stem}.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl_col2:
        # CSV: flatten for spreadsheet import
        import csv
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["field", "value"])
        for k, v in sap_payload.items():
            if k == "cooking_zones":
                for z in v:
                    writer.writerow([f"cooking_zone.{z.get('zone_id')}", z.get("diameter_mm")])
            else:
                writer.writerow([k, v])
        st.download_button(
            "⬇️ Download SAP CSV",
            data=csv_buffer.getvalue(),
            file_name=f"sap_{Path(source_path).stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )

elif status == "review_required":
    st.warning(
        "**SAP JSON not generated**\n\n"
        "**Reason:** Human review is required before ERP creation.\n\n"
        "SAP creation is paused pending human review. In future versions, "
        "SAP material master records will be created automatically after successful validation."
    )
else:  # rejected
    st.error(
        "**SAP JSON not generated**\n\n"
        "**Reason:** Processing failed. Customer action required.\n\n"
        "Downstream ERP processing has been stopped."
    )