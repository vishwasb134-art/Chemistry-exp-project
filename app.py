import io
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
from stmol import showmol

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolTransforms


# Set up the page title and layout
st.set_page_config(page_title="Molecule Checker", layout="wide")

# Add custom CSS to give the app a bright, colorful, and friendly look
st.markdown(
    """
    <style>
    /* Bright pastel laboratory background */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #eef6ff 0%, #f8fafc 50%, #ecfdf5 100%);
        color: #0f172a;
    }
    [data-testid="stHeader"] {
        background: rgba(238, 246, 255, 0.85);
    }

    /* High-contrast readable text across headings and widgets */
    h1, h2, h3, p, label, .stMarkdown, [data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
    }
    h1 {
        background: linear-gradient(90deg, #0284c7, #0d9488, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    h2, h3 {
        color: #0369a1 !important;
        border-bottom: 2px solid #bae6fd;
        padding-bottom: 0.25rem;
    }

    /* Colorful pill-style tabs */
    [data-baseweb="tab-list"] {
        background-color: #ffffff;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        gap: 8px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
    }
    [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: #334155 !important;
        font-weight: 600;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #0284c7, #0d9488) !important;
        color: #ffffff !important;
    }
    [data-baseweb="tab"][aria-selected="true"] p {
        color: #ffffff !important;
    }

    /* Bright input boxes */
    input, [data-baseweb="input"], [data-baseweb="base-input"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
    }
    [data-baseweb="input"] {
        border: 1.5px solid #93c5fd !important;
    }
    [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
        background-color: #e0f2fe !important;
        color: #0369a1 !important;
    }

    /* Colorful metric cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #bae6fd;
        border-top: 4px solid #0ea5e9;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.08);
    }
    [data-testid="stMetricLabel"] p {
        color: #0369a1 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Vibrant download buttons */
    .stDownloadButton button {
        background: linear-gradient(90deg, #0284c7, #0d9488) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 3px 8px rgba(2, 132, 199, 0.25);
    }
    .stDownloadButton button p {
        color: #ffffff !important;
    }
    .stDownloadButton button:hover {
        opacity: 0.92;
        transform: translateY(-1px);
    }

    /* Clean bright tables */
    [data-testid="stTable"] table {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stTable"] thead th {
        background-color: #e0f2fe !important;
        color: #0369a1 !important;
        font-weight: 700 !important;
    }
    [data-testid="stTable"] tbody tr:nth-child(even) {
        background-color: #f8fafc !important;
    }
    [data-testid="stTable"] tbody td {
        color: #0f172a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Molecule Checker")
st.caption("Explore 3D molecular geometry, energy optimization, Lipinski drug-likeness, and Beer-Lambert calibration.")

# Create the two tabs requested for the app
tab1, tab2 = st.tabs(["Molecule Analysis", "Beer-Lambert (Module 3)"])


# ==============================================================================
# TAB 1: MOLECULE ANALYSIS
# ==============================================================================
with tab1:
    st.header("Molecule Analysis")

    # 1. Text box for the user to enter a SMILES string (default is ethanol: CCO)
    smiles_input = st.text_input("Enter a SMILES string:", value="CCO")

    # Check if the input is empty or invalid
    smiles_clean = smiles_input.strip()
    mol = Chem.MolFromSmiles(smiles_clean) if smiles_clean else None

    if mol is None:
        st.error("Invalid SMILES string. Please enter a valid SMILES (for example: CCO, c1ccccc1, CC(=O)O).")
    else:
        # 2. Add hydrogens, generate a 3D structure, and optimize step-by-step with MMFF
        mol_3d = Chem.AddHs(mol)

        # EmbedMolecule generates 3D coordinates using ETKDG
        embed_status = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
        if embed_status != 0:
            # Fallback to random coordinates if standard embedding fails
            embed_status = AllChem.EmbedMolecule(mol_3d, useRandomCoords=True)

        if embed_status != 0:
            st.error("Could not generate a 3D structure for this molecule.")
        else:
            # Set up the MMFF94 force field
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol_3d)
            if mmff_props is None:
                st.error("MMFF force field parameters are not available for this molecule.")
            else:
                force_field = AllChem.MMFFGetMoleculeForceField(mol_3d, mmff_props)

                # Record the starting energy (step 0)
                energies = [force_field.CalcEnergy()]

                # Run 40 optimization steps, one step at a time, recording energy after each step
                num_steps = 40
                for _ in range(num_steps):
                    force_field.Minimize(maxIts=1)
                    energies.append(force_field.CalcEnergy())

                # Show 3D structure and Energy plot side by side
                col_3d, col_energy = st.columns(2)

                # 3. Interactive 3D structure using py3Dmol and stmol (stick and sphere style)
                with col_3d:
                    st.subheader("3D Structure (Optimized)")
                    show_vdw = st.checkbox("Show Van der Waals Surface", value=False)
                    mol_block = Chem.MolToMolBlock(mol_3d)
                    viewer = py3Dmol.view(width=500, height=400)
                    viewer.setBackgroundColor("#f0f7ff")
                    viewer.addModel(mol_block, "mol")
                    viewer.setStyle({
                        "stick": {"radius": 0.16, "colorscheme": "Jmol"},
                        "sphere": {"scale": 0.26, "colorscheme": "Jmol"},
                    })
                    if show_vdw:
                        viewer.addSurface(py3Dmol.VDW, {"opacity": 0.65, "colorscheme": "Jmol"})
                    viewer.zoomTo()
                    showmol(viewer, height=400, width=500)

                    # Export optimized 3D geometry as PDB or SDF
                    pdb_data = Chem.MolToPDBBlock(mol_3d)
                    sdf_buffer = io.StringIO()
                    sdf_writer = Chem.SDWriter(sdf_buffer)
                    sdf_writer.write(mol_3d)
                    sdf_writer.close()
                    sdf_data = sdf_buffer.getvalue()

                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button(
                            label="Download PDB (.pdb)",
                            data=pdb_data,
                            file_name="molecule.pdb",
                            mime="chemical/x-pdb",
                        )
                    with dl_col2:
                        st.download_button(
                            label="Download SDF (.sdf)",
                            data=sdf_data,
                            file_name="molecule.sdf",
                            mime="chemical/x-mdl-sdfile",
                        )

                # 4. Plot "Energy vs Optimization Step" using matplotlib
                with col_energy:
                    st.subheader("Energy vs Optimization Step")
                    steps = list(range(len(energies)))
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor("#ffffff")
                    ax.set_facecolor("#f8fbff")
                    ax.plot(
                        steps,
                        energies,
                        marker="o",
                        markersize=5,
                        markerfacecolor="#0ea5e9",
                        markeredgecolor="#ffffff",
                        linewidth=2.2,
                        color="#6366f1",
                    )
                    ax.fill_between(steps, energies, min(energies), color="#6366f1", alpha=0.12)
                    ax.set_title("Energy vs Optimization Step", color="#0369a1", fontweight="bold")
                    ax.set_xlabel("Optimization Step", color="#334155")
                    ax.set_ylabel("MMFF Energy (kcal/mol)", color="#334155")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.grid(True, linestyle="--", alpha=0.45, color="#94a3b8")
                    st.pyplot(fig)
                    plt.close(fig)

                # 5. Tables of optimized geometry: bond lengths (Å) and bond angles (degrees)
                st.subheader("Optimized Geometry")
                conformer = mol_3d.GetConformer()

                # Helper function to create readable atom labels like C0, O1, H2
                def get_atom_label(atom_index):
                    atom = mol_3d.GetAtomWithIdx(atom_index)
                    return f"{atom.GetSymbol()}{atom_index}"

                # Collect all bond lengths
                bond_rows = []
                for bond in mol_3d.GetBonds():
                    idx1 = bond.GetBeginAtomIdx()
                    idx2 = bond.GetEndAtomIdx()
                    length = rdMolTransforms.GetBondLength(conformer, idx1, idx2)
                    bond_rows.append({
                        "Bond": f"{get_atom_label(idx1)}-{get_atom_label(idx2)}",
                        "Length (Å)": round(length, 4),
                    })

                # Collect all bond angles (triplets i-j-k where j is the central atom)
                angle_rows = []
                for center_atom in mol_3d.GetAtoms():
                    j = center_atom.GetIdx()
                    neighbors = [neighbor.GetIdx() for neighbor in center_atom.GetNeighbors()]
                    # Pair every two unique neighbors around central atom j
                    for a_pos in range(len(neighbors)):
                        for b_pos in range(a_pos + 1, len(neighbors)):
                            i = neighbors[a_pos]
                            k = neighbors[b_pos]
                            angle = rdMolTransforms.GetAngleDeg(conformer, i, j, k)
                            angle_rows.append({
                                "Angle": f"{get_atom_label(i)}-{get_atom_label(j)}-{get_atom_label(k)}",
                                "Degrees (°)": round(angle, 2),
                            })

                col_bonds, col_angles = st.columns(2)
                with col_bonds:
                    st.write("**Bond Lengths (Å)**")
                    st.table(bond_rows, hide_index=True, height=280)

                with col_angles:
                    st.write("**Bond Angles (Degrees)**")
                    st.table(angle_rows, hide_index=True, height=280)

                # 6. Drug-likeness check using Lipinski's Rule of Five
                st.subheader("Drug-Likeness Check (Lipinski's Rule of Five)")

                # Calculate descriptors on the molecule
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)

                # Check each Lipinski rule
                mw_pass = mw < 500
                logp_pass = logp < 5
                hbd_pass = hbd <= 5
                hba_pass = hba <= 10

                # Show values as metrics and a tick or cross for each rule
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Molecular Weight", f"{mw:.2f} g/mol")
                    st.write(f"{'✅' if mw_pass else '❌'} MW < 500")
                with m2:
                    st.metric("LogP", f"{logp:.2f}")
                    st.write(f"{'✅' if logp_pass else '❌'} LogP < 5")
                with m3:
                    st.metric("H-Bond Donors", f"{hbd}")
                    st.write(f"{'✅' if hbd_pass else '❌'} Donors <= 5")
                with m4:
                    st.metric("H-Bond Acceptors", f"{hba}")
                    st.write(f"{'✅' if hba_pass else '❌'} Acceptors <= 10")

                # Show green "Drug-like" or yellow "Not drug-like" message at the end
                if mw_pass and logp_pass and hbd_pass and hba_pass:
                    st.success("Drug-like")
                else:
                    st.warning("Not drug-like")


# ==============================================================================
# TAB 2: BEER-LAMBERT (MODULE 3)
# ==============================================================================
with tab2:
    st.header("Beer-Lambert (Module 3)")

    # 1. Inputs for comma-separated concentrations, absorbances, and path length
    conc_text = st.text_input(
        "Concentrations (M, comma-separated):",
        value="0.0, 0.2, 0.4, 0.6, 0.8, 1.0",
    )
    abs_text = st.text_input(
        "Absorbances (comma-separated):",
        value="0.00, 0.15, 0.31, 0.44, 0.60, 0.75",
    )
    path_length = st.number_input(
        "Path length (cm):",
        min_value=0.0001,
        value=1.0,
        step=0.1,
    )
    unknown_abs_text = st.text_input(
        "Unknown sample absorbance (optional):",
        value="",
    )

    # Helper function to convert a comma-separated string into a numpy array of floats
    def parse_number_list(text):
        parts = [item.strip() for item in text.split(",") if item.strip() != ""]
        if not parts:
            return None
        return np.array([float(x) for x in parts], dtype=float)

    # Validate inputs and handle bad input with clear error messages
    error_message = None
    try:
        concentrations = parse_number_list(conc_text)
        absorbances = parse_number_list(abs_text)

        if concentrations is None or absorbances is None:
            error_message = "Please enter comma-separated numbers for both concentrations and absorbances."
        elif len(concentrations) != len(absorbances):
            error_message = (
                f"Number of concentrations ({len(concentrations)}) must match "
                f"the number of absorbances ({len(absorbances)})."
            )
        elif len(concentrations) < 2:
            error_message = "Please enter at least 2 data points to fit a straight line."
        elif np.all(concentrations == concentrations[0]):
            error_message = "Concentrations cannot all be the same value."
        elif path_length <= 0:
            error_message = "Path length must be greater than 0 cm."
    except ValueError:
        error_message = "Invalid number format. Please enter valid numbers separated by commas."

    if error_message:
        st.error(error_message)
    else:
        # 2. Fit a straight line (Absorbance = slope * Concentration + intercept) using numpy.polyfit
        slope, intercept = np.polyfit(concentrations, absorbances, 1)

        # Beer-Lambert law: A = ε * l * c  =>  slope = ε * l  =>  ε = slope / path_length
        molar_absorptivity = slope / path_length

        # Display molar absorptivity
        st.metric("Molar Absorptivity (ε)", f"{molar_absorptivity:.4f} M⁻¹ cm⁻¹")
        st.write(f"Best-fit equation: Absorbance = {slope:.4f} × Concentration + {intercept:.4f}")

        # Plot data points and best-fit line
        fig_bl, ax_bl = plt.subplots(figsize=(7, 4.5))
        fig_bl.patch.set_facecolor("#ffffff")
        ax_bl.set_facecolor("#f8fbff")
        ax_bl.scatter(
            concentrations,
            absorbances,
            color="#0d9488",
            edgecolors="#ffffff",
            s=80,
            label="Measured Data",
            zorder=3,
        )

        x_line = np.linspace(concentrations.min(), concentrations.max(), 100)
        y_line = slope * x_line + intercept
        ax_bl.plot(x_line, y_line, color="#f97316", linewidth=2.4, label="Best-Fit Line")

        # 3. Optional unknown sample absorbance -> estimate concentration
        unknown_clean = unknown_abs_text.strip()
        if unknown_clean != "":
            try:
                unknown_abs = float(unknown_clean)
                if slope == 0:
                    st.error("Cannot estimate concentration because the fitted line has a slope of 0.")
                else:
                    estimated_conc = (unknown_abs - intercept) / slope
                    st.metric("Estimated Unknown Concentration", f"{estimated_conc:.4f} M")

                    # Show the unknown sample on the plot as well
                    ax_bl.scatter(
                        [estimated_conc],
                        [unknown_abs],
                        color="#d946ef",
                        marker="*",
                        s=180,
                        label="Unknown Sample",
                        zorder=4,
                    )
            except ValueError:
                st.error("Invalid unknown absorbance. Please enter a single valid number.")

        ax_bl.set_title("Beer-Lambert Calibration Curve", color="#0369a1", fontweight="bold")
        ax_bl.set_xlabel("Concentration (M)", color="#334155")
        ax_bl.set_ylabel("Absorbance", color="#334155")
        ax_bl.spines["top"].set_visible(False)
        ax_bl.spines["right"].set_visible(False)
        ax_bl.legend()
        ax_bl.grid(True, linestyle="--", alpha=0.45, color="#94a3b8")
        st.pyplot(fig_bl)
        plt.close(fig_bl)