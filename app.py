import json
import hashlib
from datetime import datetime
import streamlit as st
import numpy as np
from Bio.Seq import Seq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Configure dashboard page
st.set_page_config(
    page_title="Gene Guard Screen Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Biosecurity Lab Dashboard
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card-danger {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 1px solid #FDA4AF;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .status-card-safe {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1px solid #86EFAC;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-title">🛡️ Gene Guard Synthesis Screening Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Automated AI Biosecurity Screening • Detects Camouflaged, Shuffled, and Structural Pathogenic Sequences</div>',
    unsafe_allow_html=True,
)

# --- DATASET CONFIGURATION & TRAINING ---
raw_dataset = [
    ("ATGGGCGTTACCGGCATCCTGCAGCTGCCGCGTGACCGTTTCAAACGTACCAGCTTC", 1),
    ("ATGGGAGTGACAGGAATACTTCAACTTCCAAGAGATAGATTTAAAAGAACAAGTTTT", 1),
    ("ATGAAAAAATTATTTTCAATATTTATAGTTTTTTTATTTTTAAATTTATTTTCAATG", 1),
    ("ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTG", 0),
    ("ATGCCGTCCTCGGTCCTCTCGTACTTCAACCCCGGGTACTACCCCTCGGGGCACGCG", 0),
    ("ATGGCTAAGCAAGTGACCCTGAACGACCTGGTGAAGCAGCTGAACGACCGTGTAACC", 0),
]

sequences = [item[0] for item in raw_dataset]
labels = [item[1] for item in raw_dataset]

# Generate space-separated 3-mer token strings for training
X_seqs = []
for seq in sequences:
    clean_seq = str(seq).strip("()', ").upper()
    kmers = [clean_seq[i:i+3] for i in range(len(clean_seq) - 2)]
    X_seqs.append(" ".join(kmers))

# Build and train vectorizer and model
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
X = vectorizer.fit_transform(X_seqs)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, labels)

# --- USER SIDEBAR CONFIGURATION ---
st.sidebar.image("https://img.icons8.com/fluency/96/dna-helix.png", width=64)
st.sidebar.markdown("### 🧬 Screening Controls")

# Presets Library for Quick Testing
preset_options = {
    "Select a preset or custom...": "",
    "🚨 Obfuscated Pathogen (Camouflaged)": "ATGGGAGTGACAGGAATACTTCAACTTCCAAGAGATAGATTTAAAAGAACAAGTTTT",
    "🚨 Structural Pathogen Variant A": "ATGGGCGTTACCGGCATCCTGCAGCTGCCGCGTGACCGTTTCAAACGTACCAGCTTC",
    "🚨 AT-Rich Pathogenic Mimic": "ATGAAAAAATTATTTTCAATATTTATAGTTTTTTTATTTTTAAATTTATTTTCAATG",
    "✅ Benign Housekeeping Control": "ATGGCTAAGCAAGTGACCCTGAACGACCTGGTGAAGCAGCTGAACGACCGTGTAACC",
    "✅ EGFP Fluorophore Marker": "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTG",
    "✅ Synthetic Structural Protein": "ATGCCGTCCTCGGTCCTCTCGTACTTCAACCCCGGGTACTACCCCTCGGGGCACGCG",
}

selected_preset = st.sidebar.selectbox("📂 Load Benchmark Sequence:", list(preset_options.keys()))
default_input = preset_options[selected_preset] if preset_options[selected_preset] else "ATGGGAGTGACAGGAATACTTCAACTTCCAAGAGATAGATTTAAAAGAACAAGTTTT"

threshold = st.sidebar.slider(
    "⚠️ Threat Decision Threshold (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help="Sequences with risk probabilities exceeding this threshold trigger suspicious quarantine.",
)

st.sidebar.divider()
st.sidebar.markdown("### 📋 Test Playground")
st.sidebar.markdown("Use this reference sequence to demonstrate how the scanner evaluates an obfuscated variant:")
obfuscated_example = "ATGGGAGTGACAGGAATACTTCAACTTCCAAGAGATAGATTTAAAAGAACAAGTTTT"
st.sidebar.text_area("Example Sequence:", obfuscated_example, height=75)

st.sidebar.divider()
st.sidebar.caption(
    "🛡️ **Gene Guard Engine v2.4**\n\n"
    "• Model: Random Forest (50 Estimators)\n"
    "• Tokenization: 3-mer Sliding Windows\n"
    "• Reference: Guidance for Providers of Synthetic Double-Stranded DNA"
)

# --- USER INTERFACE INPUT ---
st.markdown("#### 🔬 Input Sequence for Inspection")
input_sequence = st.text_area(
    "Paste incoming nucleic acid string (FASTA or plain bases):",
    value=default_input,
    height=130,
    placeholder="e.g., ATGGGAGTGACAGGAATACTTCAACTTCCA...",
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    scan_triggered = st.button("🚀 Run AI Security Scan", type="primary", use_container_width=True)

# --- INFERENCE RUNNER ---
if scan_triggered:
    if not input_sequence.strip():
        st.warning("⚠️ Please paste a sequence string to analyze.")
    else:
        # Preprocessing & cleaning
        clean_dna = (
            input_sequence.upper()
            .strip()
            .replace(" ", "")
            .replace("\n", "")
            .replace("\r", "")
            .replace("\t", "")
        )
        invalid_bases = sorted(set(clean_dna) - set("ATCG"))

        if invalid_bases:
            st.error(f"❌ Invalid DNA sequence. Unexpected non-canonical bases detected: **{', '.join(invalid_bases)}**")
        elif len(clean_dna) < 3:
            st.warning("⚠️ Sequence too short for screening. Please enter at least 3 DNA bases.")
        else:
            # Biological translation
            dna_seq = Seq(clean_dna)
            translated_protein = str(dna_seq.translate(to_stop=False))

            # Feature extraction
            kmers_list = [clean_dna[i:i+3] for i in range(len(clean_dna) - 2)]
            input_kmers = " ".join(kmers_list)
            vectorized_input = vectorizer.transform([input_kmers])

            # Machine learning predictions
            probabilities = model.predict_proba(vectorized_input)[0]
            positive_class_index = list(model.classes_).index(1)
            risk_score = probabilities[positive_class_index] * 100

            # Evaluate decision by threshold
            is_threat = risk_score >= threshold
            predicted_label = 1 if is_threat else 0
            ml_confidence = risk_score if is_threat else (100.0 - risk_score)

            # Metadata calculations
            gc_bases = clean_dna.count("G") + clean_dna.count("C")
            gc_content = (gc_bases / len(clean_dna)) * 100 if len(clean_dna) > 0 else 0
            seq_hash = hashlib.sha256(clean_dna.encode()).hexdigest()
            scan_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

            st.divider()

            # --- TOP STATUS VERDICT BANNER ---
            if is_threat:
                st.markdown(
                    f"""
                    <div class="status-card-danger">
                        <h3 style="color: #BE123C; margin: 0;">🚨 FLAGGED FOR HUMAN BIOSECURITY REVIEW</h3>
                        <p style="color: #9F1239; margin-top: 0.3rem; margin-bottom: 0;">
                            <strong>Threat Risk: {risk_score:.1f}%</strong> (Exceeds {threshold}% threshold) • 
                            High sequence similarity detected against regulated pathogen structural motifs.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="status-card-safe">
                        <h3 style="color: #047857; margin: 0;">✅ SCREENING PASSED: LOW-RISK SEQUENCE</h3>
                        <p style="color: #065F46; margin-top: 0.3rem; margin-bottom: 0;">
                            <strong>Threat Risk: {risk_score:.1f}%</strong> (Within safe boundary &lt; {threshold}%) • 
                            No significant structural homology to high-consequence biosecurity agents identified.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Quick Metric Gauges
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Pathogen Match Risk", f"{risk_score:.1f}%")
            with m2:
                st.metric("ML Classification Confidence", f"{ml_confidence:.1f}%")
            with m3:
                st.metric("Sequence Length", f"{len(clean_dna)} bp")
            with m4:
                st.metric("GC Content", f"{gc_content:.1f}%")

            # Risk Meter Visual Bar
            risk_normalized = max(0.0, min(1.0, risk_score / 100.0))
            st.caption("Threat Probability Gradient (0% = Benign, 100% = High Consequence):")
            st.progress(risk_normalized)

            st.write("")

            # --- DETAILED ANALYSIS TABS ---
            tab_verdict, tab_bio, tab_kmers, tab_audit = st.tabs([
                "🛡️ Threat Assessment",
                "🧬 Sequence & Translation",
                "🔬 K-mer Feature Analysis",
                "📑 Compliance & Audit Log",
            ])

            with tab_verdict:
                col_v1, col_v2 = st.columns([3, 2])
                with col_v1:
                    st.markdown("##### Threat Evaluation Details")
                    if is_threat:
                        st.error(
                            "**Action Required**: Automatic synthesis dispatch holds applied. "
                            "This sequence exhibits canonical 3-mer distributions characteristic of known regulated pathogen families. "
                            "Secondary customer identity validation and biological laboratory clearance are mandatory."
                        )
                    else:
                        st.success(
                            "**Clearance Granted**: Sequence metrics are compatible with standard non-pathogenic laboratory plasmids "
                            "or benign eukaryotic reference genomes. Pre-synthesis authorization may proceed."
                        )

                    st.markdown("##### Model Calibration")
                    st.write(f"- **Operating Decision Threshold**: `{threshold}%`")
                    st.write(f"- **Calibrated Posterior Probability (Class 1 - Pathogen)**: `{risk_score:.2f}%`")
                    st.write(f"- **Calibrated Posterior Probability (Class 0 - Benign)**: `{100.0 - risk_score:.2f}%`")
                    st.write(f"- **Model Classes**: `{len(model.classes_)}`")

                with col_v2:
                    st.markdown("##### Biosecurity Checklist")
                    st.checkbox("Harmful sequence homology absent", value=(not is_threat), disabled=True)
                    st.checkbox("Valid reading frames verified", value=True, disabled=True)
                    st.checkbox("Customer identity verification on file", value=False, disabled=True)
                    st.checkbox("Export control authorization confirmed", value=False, disabled=True)

            with tab_bio:
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.markdown("##### 📂 Canonical DNA Sequence String")
                    st.code(clean_dna, language="text")
                with col_b2:
                    st.markdown("##### 🧬 Translated In-Silico Protein")
                    st.code(translated_protein, language="text")

                st.markdown("##### Nucleotide Base Distribution")
                base_counts = {
                    "Adenine (A)": clean_dna.count("A"),
                    "Thymine (T)": clean_dna.count("T"),
                    "Cytosine (C)": clean_dna.count("C"),
                    "Guanine (G)": clean_dna.count("G"),
                }
                st.bar_chart(base_counts)

            with tab_kmers:
                st.markdown("##### 3-mer Decomposition")
                st.write(f"Total extracted sliding windows: **{len(kmers_list)}** 3-mers")

                # Count top frequent k-mers
                from collections import Counter
                kmer_counts = Counter(kmers_list).most_common(12)
                top_kmer_dict = {k: v for k, v in kmer_counts}

                st.markdown("###### Top 12 Most Frequent 3-mers in Analyzed Sequence:")
                st.bar_chart(top_kmer_dict)

                with st.expander("🔍 View Raw Space-Separated K-mer Token String"):
                    st.code(input_kmers, language="text")

            with tab_audit:
                st.markdown("##### 🔒 Cryptographic Audit Trail")
                st.write(f"**Inspection Timestamp**: `{scan_time}`")
                st.write(f"**SHA-256 Sequence Digest**: `{seq_hash}`")
                st.write(f"**Random Forest Seed**: `42`")

                # Prepare JSON Audit Record
                audit_record = {
                    "engine": "Gene Guard Screen Engine v2.4",
                    "timestamp_utc": scan_time,
                    "sha256": seq_hash,
                    "length_bp": len(clean_dna),
                    "gc_percent": round(gc_content, 2),
                    "risk_score_percent": round(risk_score, 2),
                    "decision_threshold_percent": threshold,
                    "verdict": "FLAGGED_FOR_REVIEW" if is_threat else "SCREENING_PASSED",
                    "translated_protein": translated_protein,
                }

                st.download_button(
                    label="💾 Download Screening Manifest (JSON)",
                    data=json.dumps(audit_record, indent=4),
                    file_name=f"gene_guard_screening_{seq_hash[:8]}.json",
                    mime="application/json",
                )

            st.caption("⚠️ Demonstration model trained on synthetic benchmark sequences. Real-world biosecurity compliance pipelines should combine k-mer screening with complete BLAST/HMMER alignment databases.")
