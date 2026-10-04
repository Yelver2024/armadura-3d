import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="Analizador Avanzado de Armaduras 3D", page_icon="🏗️", layout="wide")

st.title("🏗️ Simulador Avanzado - Armaduras 3D")
st.markdown("### Método de Rigidez Directo | Visor Gráfico 3D y Memoria de Cálculo")

# --- FUNCIONES AUXILIARES PARA LATEX PURO (EN PANTALLA) ---
def matriz_a_latex(mat, decimales=2):
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    filas_texto = []
    for fila in mat:
        elementos_fila = " & ".join([f"{val:.{decimales}f}" for val in fila])
        filas_texto.append(elementos_fila)
    return "\\begin{bmatrix}\n" + " \\\\\n".join(filas_texto) + "\n\\end{bmatrix}"

def mostrar_matriz_con_gdl_latex(mat, gdl_labels, nombre_matriz="", unidades=""):
    n = len(gdl_labels)
    gdl_fila_str = " & ".join([f"\\mathbf{{{lbl}}}" for lbl in gdl_labels])
    
    filas_matriz = []
    for fila in mat:
        filas_matriz.append(" & ".join([f"{val:.6f}" if val < 1 else f"{val:.2f}" for val in fila]))
    cuerpo_matriz = " \\\\\n".join(filas_matriz)
    
    gdl_col = "\\begin{matrix}\n" + " \\\\\n".join([f"\\mathbf{{{lbl}}}" for lbl in gdl_labels]) + "\n\\end{matrix}"
    columnas_formato = 'r' * n
    
    formula = (
        f"{nombre_matriz} \\; [{unidades}] = "
        f"\\begin{{array}}{{c}}"
        f"\\begin{{array}}{{{columnas_formato}}}"
        f"{gdl_fila_str}"
        f"\\end{{array}} \\\\"
        f"\\left[ \\begin{{array}}{{{columnas_formato}}}"
        f"{cuerpo_matriz}"
        f"\\end{{array}} \\right]"
        f"\\end{{array}}"
        f"\\quad {gdl_col}"
    )
    st.latex(formula)

def vector_con_gdl_latex(vec, gdl_labels, decimales=5):
    v_lat = matriz_a_latex(vec, decimales=decimales)
    gdl_col = "\\begin{matrix}\n" + " \\\\\n".join([f"\\mathbf{{{lbl}}}" for lbl in gdl_labels]) + "\n\\end{matrix}"
    return f"{v_lat} \\quad {gdl_col}"

def expresion_matriz_con_gdl_superior(mat, gdl_labels, decimales=6):
    n = len(gdl_labels)
    gdl_fila_str = " & ".join([f"\\mathbf{{{lbl}}}" for lbl in gdl_labels])
    filas_matriz = []
    for fila in mat:
        filas_matriz.append(" & ".join([f"{val:.{decimales}f}" for val in fila]))
    cuerpo_matriz = " \\\\\n".join(filas_matriz)
    gdl_col = "\\begin{matrix}\n" + " \\\\\n".join([f"\\mathbf{{{lbl}}}" for lbl in gdl_labels]) + "\n\\end{matrix}"
    columnas_formato = 'r' * n
    
    mat_latex = (
        f"\\begin{{array}}{{c}}"
        f"\\begin{{array}}{{{columnas_formato}}}"
        f"{gdl_fila_str}"
        f"\\end{{array}} \\\\"
        f"\\left[ \\begin{{array}}{{{columnas_formato}}}"
        f"{cuerpo_matriz}"
        f"\\end{{array}} \\right]"
        f"\\end{{array}}"
        f"\\quad {gdl_col}"
    )
    return mat_latex

# --- FUNCIÓN PARA CREAR MATRICES REALES CON CORCHETES EN PDF ---
def crear_tabla_matriz_con_corchetes(matriz, decimales=2, max_filas=15):
    if matriz.ndim == 1:
        matriz = matriz.reshape(-1, 1)
    
    filas_mostrar = min(len(matriz), max_filas)
    datos_internos = []
    for i in range(filas_mostrar):
        fila_strs = [f"{val:.{decimales}f}" for val in matriz[i]]
        datos_internos.append(fila_strs)
    
    if len(matriz) > max_filas:
        datos_internos.append(["..." for _ in range(matriz.shape[1])])

    # Agregamos columnas vacías a los extremos izquierdo y derecho para los corchetes
    n_cols = len(datos_internos[0])
    matriz_completa = []
    for fila in datos_internos:
        matriz_completa.append([""] + fila + [""])

    t = Table(matriz_completa, hAlign='CENTER')
    
    # Estilo que dibuja los corchetes reales en las columnas de los extremos [ y ]
    estilo_tabla = [
        ('FONTNAME', (1,0), (-2,-1), 'Helvetica'),
        ('FONTSIZE', (1,0), (-2,-1), 7),
        ('ALIGN', (1,0), (-2,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        
        # Corchete izquierdo (Línea vertical y esquinas superior/inferior)
        ('BOX', (0,0), (0,-1), 1, colors.black),
        ('LINEABOVE', (0,0), (0,0), 1.5, colors.black),
        ('LINEBELOW', (0,-1), (0,-1), 1.5, colors.black),
        
        # Corchete derecho (Línea vertical y esquinas superior/inferior)
        ('BOX', (-1,0), (-1,-1), 1, colors.black),
        ('LINEABOVE', (-1,0), (-1,0), 1.5, colors.black),
        ('LINEBELOW', (-1,-1), (-1,-1), 1.5, colors.black),
    ]
    
    t.setStyle(TableStyle(estilo_tabla))
    return t

# --- BARRA LATERAL CON FORMULARIO ESTABLE ---
with st.sidebar:
    st.header("🛠 Configuración de la Estructura")
    with st.form(key="form_configuracion_armadura"):
        st.subheader("1. Coordenadas de Nodos y Apoyos")
        df_nodos_default = pd.DataFrame([
            {"Nodo": 1, "X": -4.0, "Y": -3.0, "Z": 0.0, "Apoyo_X": True, "Apoyo_Y": True, "Apoyo_Z": True},
            {"Nodo": 2, "X": 4.0, "Y": -3.0, "Z": 0.0, "Apoyo_X": True, "Apoyo_Y": True, "Apoyo_Z": True},
            {"Nodo": 3, "X": 4.0, "Y": 3.0, "Z": 0.0, "Apoyo_X": True, "Apoyo_Y": True, "Apoyo_Z": True},
            {"Nodo": 4, "X": -4.0, "Y": 3.0, "Z": 0.0, "Apoyo_X": True, "Apoyo_Y": True, "Apoyo_Z": True},
            {"Nodo": 5, "X": 0.0, "Y": 0.0, "Z": 10.0, "Apoyo_X": False, "Apoyo_Y": False, "Apoyo_Z": False},
        ])
        df_nodos = st.data_editor(df_nodos_default, num_rows="dynamic", key="editor_nodos_estable")

        st.subheader("2. Conectividad de Elementos (Barras)")
        df_barras_default = pd.DataFrame([
            {"Barra": 1, "Inicio": 4, "Fin": 5, "Area_m2": 0.001, "E_KN_m2": 200000000.0},
            {"Barra": 2, "Inicio": 5, "Fin": 2, "Area_m2": 0.001, "E_KN_m2": 200000000.0},
            {"Barra": 3, "Inicio": 5, "Fin": 3, "Area_m2": 0.001, "E_KN_m2": 200000000.0},
            {"Barra": 4, "Inicio": 5, "Fin": 1, "Area_m2": 0.001, "E_KN_m2": 200000000.0},
        ])
        df_barras = st.data_editor(df_barras_default, num_rows="dynamic", key="editor_barras_estable")

        st.subheader("3. Cargas Nodales (KN)")
        df_cargas_default = pd.DataFrame([
            {"Nodo": 5, "Fx_KN": 60.0, "Fy_KN": -80.0, "Fz_KN": 0.0},
            {"Nodo": 1, "Fx_KN": 0.0, "Fy_KN": 0.0, "Fz_KN": 0.0},
            {"Nodo": 2, "Fx_KN": 0.0, "Fy_KN": 0.0, "Fz_KN": 0.0},
            {"Nodo": 3, "Fx_KN": 0.0, "Fy_KN": 0.0, "Fz_KN": 0.0},
            {"Nodo": 4, "Fx_KN": 0.0, "Fy_KN": 0.0, "Fz_KN": 0.0},
        ])
        df_cargas = st.data_editor(df_cargas_default, num_rows="dynamic", key="editor_cargas_estable")

        submit_button = st.form_submit_button(label="🔄 Refrescar y Calcular Estructura", type="primary")

# --- PROCESAMIENTO Y CÁLCULO ---
try:
    nodos = {}
    restricciones = {}
    for _, row in df_nodos.iterrows():
        nid = int(row["Nodo"])
        nodos[nid] = np.array([row["X"], row["Y"], row["Z"]], dtype=float)
        restricciones[nid] = [bool(row["Apoyo_X"]), bool(row["Apoyo_Y"]), bool(row["Apoyo_Z"])]

    elementos = []
    for _, row in df_barras.iterrows():
        elementos.append({
            "id": int(row["Barra"]),
            "ini": int(row["Inicio"]),
            "fin": int(row["Fin"]),
            "A": float(row["Area_m2"]),
            "E": float(row["E_KN_m2"])
        })

    cargas_nodales = {}
    for _, row in df_cargas.iterrows():
        nid = int(row["Nodo"])
        fvec = np.array([row["Fx_KN"], row["Fy_KN"], row["Fz_KN"]], dtype=float)
        if np.any(fvec != 0):
            cargas_nodales[nid] = fvec

    # --- ASIGNACIÓN DE GDL ---
    n_nodos = len(nodos)
    n_gdl = n_nodos * 3
    nodo_ids = sorted(list(nodos.keys()))

    nodos_libres = [nid for nid in nodo_ids if not any(restricciones[nid])]
    nodos_restringidos = [nid for nid in nodo_ids if any(restricciones[nid])]
    nodo_ids_ordenados = nodos_libres + nodos_restringidos
    
    nodo_a_gdl_indices = {}
    gdl_global_labels = []
    
    for idx_gdl_base, nid in enumerate(nodo_ids_ordenados):
        inicio_gdl = idx_gdl_base * 3
        nodo_a_gdl_indices[nid] = [inicio_gdl, inicio_gdl + 1, inicio_gdl + 2]
        gdl_global_labels.extend([
            f"{inicio_gdl + 1}\\text{{g}}",
            f"{inicio_gdl + 2}\\text{{g}}",
            f"{inicio_gdl + 3}\\text{{g}}"
        ])

    # --- CÁLCULO MATRICIAL ---
    K_global = np.zeros((n_gdl, n_gdl))
    F_global = np.zeros(n_gdl)

    for nodo_id, f_vec in cargas_nodales.items():
        if nodo_id in nodo_a_gdl_indices:
            indices = nodo_a_gdl_indices[nodo_id]
            F_global[indices] = f_vec

    matrices_elementos = []
    for el in elementos:
        i, j = el["ini"], el["fin"]
        if i not in nodos or j not in nodos:
            continue
        A, E = el["A"], el["E"]
        Xi, Yi, Zi = nodos[i]
        Xj, Yj, Zj = nodos[j]

        dx, dy, dz = Xj - Xi, Yj - Yi, Zj - Zi
        L = np.sqrt(dx**2 + dy**2 + dz**2)
        if L == 0:
            continue
        cx, cy, cz = dx / L, dy / L, dz / L
        k_axial = (E * A) / L

        c_vec = np.array([cx, cy, cz, -cx, -cy, -cz])
        k_elem = k_axial * np.outer(c_vec, c_vec)
        
        gdl_i_indices = nodo_a_gdl_indices[i]
        gdl_j_indices = nodo_a_gdl_indices[j]
        gdl_elem_indices = gdl_i_indices + gdl_j_indices
        
        gdl_elem_labels = [
            f"{gdl_i_indices[0]+1}\\text{{g}}", f"{gdl_i_indices[1]+1}\\text{{g}}", f"{gdl_i_indices[2]+1}\\text{{g}}",
            f"{gdl_j_indices[0]+1}\\text{{g}}", f"{gdl_j_indices[1]+1}\\text{{g}}", f"{gdl_j_indices[2]+1}\\text{{g}}"
        ]

        matrices_elementos.append({
            "id": el["id"], "ini": i, "fin": j, "L": L, "cx": cx, "cy": cy, "cz": cz, "k": k_elem, "gdl_labels": gdl_elem_labels, "E": E, "A": A
        })

        for r in range(6):
            for c in range(6):
                K_global[gdl_elem_indices[r], gdl_elem_indices[c]] += k_elem[r, c]

    gdl_libres = []
    gdl_restringidos = []
    for nid in nodo_ids_ordenados:
        indices = nodo_a_gdl_indices[nid]
        rest = restricciones[nid]
        for k in range(3):
            if rest[k]:
                gdl_restringidos.append(indices[k])
            else:
                gdl_libres.append(indices[k])

    if len(gdl_libres) == 0:
        st.error("Error: Todos los grados de libertad están restringidos. Libera al menos un nodo.")
    else:
        K_LL = K_global[np.ix_(gdl_libres, gdl_libres)]
        K_LR = K_global[np.ix_(gdl_libres, gdl_restringidos)]
        K_RL = K_global[np.ix_(gdl_restringidos, gdl_libres)]
        F_LL = F_global[gdl_libres]
        
        K_LL_inv = np.linalg.inv(K_LL)
        D_libres = K_LL_inv @ F_LL
        
        D_global = np.zeros(n_gdl)
        D_global[gdl_libres] = D_libres

        F_RR = K_RL @ D_libres
        R_global = np.zeros(n_gdl)
        R_global[gdl_restringidos] = F_RR

        gdl_libres_labels = [gdl_global_labels[idx] for idx in gdl_libres]
        gdl_restringidos_labels = [gdl_global_labels[idx] for idx in gdl_restringidos]

        resultados_fuerzas = {}
        for el in elementos:
            i, j = el["ini"], el["fin"]
            if i not in nodos or j not in nodos:
                continue
            Xi, Yi, Zi = nodos[i]
            Xj, Yj, Zj = nodos[j]
            dx, dy, dz = Xj - Xi, Yj - Yi, Zj - Zi
            L = np.sqrt(dx**2 + dy**2 + dz**2)
            cx, cy, cz = dx/L, dy/L, dz/L

            gdl_i_indices = nodo_a_gdl_indices[i]
            gdl_j_indices = nodo_a_gdl_indices[j]
            gdl_elem_indices = gdl_i_indices + gdl_j_indices
            gdl_elem_labels = [
                f"{gdl_i_indices[0]+1}\\text{{g}}", f"{gdl_i_indices[1]+1}\\text{{g}}", f"{gdl_i_indices[2]+1}\\text{{g}}",
                f"{gdl_j_indices[0]+1}\\text{{g}}", f"{gdl_j_indices[1]+1}\\text{{g}}", f"{gdl_j_indices[2]+1}\\text{{g}}"
            ]

            d_elem = np.array([
                D_global[gdl_i_indices[0]], D_global[gdl_i_indices[1]], D_global[gdl_i_indices[2]],
                D_global[gdl_j_indices[0]], D_global[gdl_j_indices[1]], D_global[gdl_j_indices[2]]
            ])
            
            factor = (el["E"] * el["A"] / L)
            matriz_geom = factor * np.array([[-cx, -cy, -cz, cx, cy, cz]])
            
            fuerza_axial = float((matriz_geom @ d_elem).item())
            estado = "Tensión (T)" if fuerza_axial >= 0 else "Compresión (C)"
            
            resultados_fuerzas[el['id']] = {
                "ini": i, "fin": j, "fval": fuerza_axial, "estado": estado,
                "L": L, "matriz_geom": matriz_geom, "d_elem": d_elem, "gdl_labels": gdl_elem_labels
            }

        # --- SISTEMA DE PESTAÑAS ---
        tab1, tab2, tab3 = st.tabs(["📊 Resultados y Gráficos", "📝 Procedimiento Detallado", "📄 Generar Reporte PDF"])

        with tab1:
            st.subheader("🌐 Visualización Gráfica 3D de la Estructura y GDL")
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                fig_prev = plt.figure(figsize=(6, 5))
                ax_prev = fig_prev.add_subplot(111, projection='3d')
                for el in elementos:
                    if el["ini"] in nodos and el["fin"] in nodos:
                        p1 = nodos[el["ini"]]
                        p2 = nodos[el["fin"]]
                        ax_prev.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='blue', linewidth=2)
                        pm_x, pm_y, pm_z = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2
                        ax_prev.text(pm_x, pm_y, pm_z, f"B{el['id']}", color='darkblue', fontsize=8, fontweight='bold')
                for nid, coord in nodos.items():
                    is_apoyo = any(restricciones[nid])
                    if is_apoyo:
                        ax_prev.scatter(coord[0], coord[1], coord[2], color='black', marker='s', s=100)
                        ax_prev.text(coord[0], coord[1], coord[2], f' N{nid}', fontsize=9, fontweight='bold', color='black')
                    else:
                        ax_prev.scatter(coord[0], coord[1], coord[2], color='red', marker='o', s=100)
                        ax_prev.text(coord[0], coord[1], coord[2], f' N{nid}', fontsize=9, fontweight='bold', color='red')
                ax_prev.set_title("Estructura 3D Interactivo")
                st.pyplot(fig_prev)

            with col_v2:
                fig_gdl = plt.figure(figsize=(6, 5))
                ax_gdl = fig_gdl.add_subplot(111, projection='3d')
                for el in elementos:
                    if el["ini"] in nodos and el["fin"] in nodos:
                        p1 = nodos[el["ini"]]
                        p2 = nodos[el["fin"]]
                        ax_gdl.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', linestyle='--', linewidth=1, alpha=0.5)
                for nid, coord in nodos.items():
                    gdl_inds = nodo_a_gdl_indices[nid]
                    gdl_text = f"N{nid}\n[{gdl_inds[0]+1}g, {gdl_inds[1]+1}g, {gdl_inds[2]+1}g]"
                    is_apoyo = any(restricciones[nid])
                    color_nodo = 'black' if is_apoyo else 'magenta'
                    ax_gdl.scatter(coord[0], coord[1], coord[2], color=color_nodo, s=120)
                    ax_gdl.text(coord[0], coord[1], coord[2] + 0.4, gdl_text, fontsize=9, fontweight='bold', color=color_nodo)
                ax_gdl.set_title("Asignación de GDL")
                st.pyplot(fig_gdl)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📍 Desplazamientos Nodales")
                for nid in nodo_ids:
                    base = nodo_a_gdl_indices[nid]
                    st.write(f"**N{nid}** -> dX:{D_global[base[0]]:.5f}m | dY:{D_global[base[1]]:.5f}m | dZ:{D_global[base[2]]:.5f}m")
                
                st.subheader("⚡ Reacciones en Apoyos")
                for nid in nodo_ids:
                    base = nodo_a_gdl_indices[nid]
                    rx, ry, rz = R_global[base[0]], R_global[base[1]], R_global[base[2]]
                    if any(restricciones[nid]):
                        st.write(f"**Apoyo N{nid}** -> Rx:{rx:.2f} KN | Ry:{ry:.2f} KN | Rz:{rz:.2f} KN")

            with col2:
                st.subheader("📊 Fuerzas Axiales en Barras")
                for bid, data in resultados_fuerzas.items():
                    st.write(f"**Barra {bid}** (N{data['ini']}→N{data['fin']}): `{data['fval']:.2f} KN` [{data['estado']}]")

        with tab2:
            st.subheader("📋 Memoria Descriptiva y Procedimiento Detallado")
            
            st.markdown("### 1. Cosenos Directores ($\lambda_x, \lambda_y, \lambda_z$) de Cada Elemento")
            for m in matrices_elementos:
                st.write(f"**Barra {m['id']} (N{m['ini']} $\\to$ N{m['fin']})** | Longitud $L = {m['L']:.3f}$ m")
                st.latex(f"\\lambda_x = {m['cx']:.4f}, \\quad \\lambda_y = {m['cy']:.4f}, \\quad \\lambda_z = {m['cz']:.4f}")

            st.markdown("### 2. Matriz de Rigidez Global de Cada Elemento ($6 \\times 6$)")
            for m in matrices_elementos:
                st.write(f"**Cálculo para Elemento {m['id']} (N{m['ini']} $\\to$ N{m['fin']}):**")
                st.latex(f"\\frac{{E \\cdot A}}{{L}} = \\frac{{{m['E']} \\times {m['A']}}}{{{m['L']:.3f}}} = { (m['E']*m['A'])/m['L']:.2f} \\text{{ KN/m}}")
                mostrar_matriz_con_gdl_latex(m['k'], m['gdl_labels'], nombre_matriz=f"K_{{global(Elemento {m['id']})}}", unidades="KN/m")

            st.markdown("### 3. Matriz Global del Sistema de la Armadura Espacial ($K_{global}$)")
            mostrar_matriz_con_gdl_latex(K_global, gdl_global_labels, nombre_matriz="K_{global}", unidades="KN/m")

            st.markdown("### 4. Ley de Hooke y Partición de la Matriz ($K_{LL}$)")
            mostrar_matriz_con_gdl_latex(K_LL, gdl_libres_labels, nombre_matriz="K_{LL}", unidades="KN/m")
            
            st.markdown("### 5. Cálculo de Desplazamientos Nodales ($D_L = K_{LL}^{-1} \\cdot F_{LL}$)")
            mat_inv_completa = expresion_matriz_con_gdl_superior(K_LL_inv, gdl_libres_labels, decimales=6)
            vec_f_completo = vector_con_gdl_latex(F_LL, gdl_libres_labels, decimales=2)
            st.latex(f"D_L = K_{{LL}}^{{-1}} \\cdot F_{{LL}} \\quad \\Rightarrow \\quad D_L = {mat_inv_completa} \\cdot {vec_f_completo}")
            
            vec_d_resultado = vector_con_gdl_latex(D_libres, gdl_libres_labels, decimales=6)
            st.latex(f"D_L = {vec_d_resultado}")

            st.markdown("### 6. Cálculo de Reacciones en Apoyos ($R_R = K_{RL} \\cdot D_L$)")
            n_cols_rl = len(gdl_libres_labels)
            gdl_fila_rl = " & ".join([f"\\mathbf{{{lbl}}}" for lbl in gdl_libres_labels])
            gdl_col_rl = "\\begin{matrix}\n" + " \\\\\n".join([f"\\mathbf{{{lbl}}}" for lbl in gdl_restringidos_labels]) + "\n\\end{matrix}"
            mat_rl_completa = (
                f"\\begin{{array}}{{c}}"
                f"\\begin{{array}}{{{'r'*n_cols_rl}}}"
                f"{gdl_fila_rl}"
                f"\\end{{array}} \\\\"
                f"\\left[ \\begin{{array}}{{{'r'*n_cols_rl}}}"
                f"{" \\\\\n".join([" & ".join([f"{val:.2f}" for val in fila]) for fila in K_RL])}"
                f"\\end{{array}} \\right]"
                f"\\end{{array}}"
                f"\\quad {gdl_col_rl}"
            )
            vec_d_completo = vector_con_gdl_latex(D_libres, gdl_libres_labels, decimales=6)
            st.latex(f"R_R = K_{{RL}} \\cdot D_L \\quad \\Rightarrow \\quad R_R = {mat_rl_completa} \\cdot {vec_d_completo}")
            
            vec_r_resultado = vector_con_gdl_latex(F_RR, gdl_restringidos_labels, decimales=2)
            st.latex(f"R_R = {vec_r_resultado}")

            st.markdown("### 7. Determinación de Fuerzas Axiales Mediante Multiplicación de Matrices")
            st.latex("N_i = \\left( \\frac{E \\cdot A}{L} \\begin{bmatrix} -\\lambda_x & -\\lambda_y & -\\lambda_z & \\lambda_x & \\lambda_y & \\lambda_z \\end{bmatrix} \\right) \\cdot \\begin{bmatrix} d_{ix} \\\\ d_{iy} \\\\ d_{iz} \\\\ d_{jx} \\\\ d_{jy} \\\\ d_{jz} \\end{bmatrix}")
            for bid, data in resultados_fuerzas.items():
                st.write(f"**Barra {bid} (N{data['ini']} $\\to$ N{data['fin']})**")
                vec_d_con_gdl = vector_con_gdl_latex(data['d_elem'], data['gdl_labels'], decimales=5)
                st.latex(f"N_{{B{bid}}} = {matriz_a_latex(data['matriz_geom'], decimales=2)} \\cdot {vec_d_con_gdl} = \\mathbf{{{data['fval']:.2f}\\text{{ KN}}}} \\quad \\rightarrow \\quad \\text{{{data['estado']}}}")

        with tab3:
            st.subheader("📄 Generación de Reporte Académico en PDF")
            st.markdown("Haz clic en el botón para compilar la Memoria de Cálculo completa en formato PDF con matrices reales enmarcadas entre corchetes, gráficos 3D y resultados.")
            
            if st.button("🖨 Compilar y Descargar Reporte PDF"):
                pdf_path = "Memoria_Calculo_Armadura_3D.pdf"
                doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
                story = []
                styles = getSampleStyleSheet()

                # --- GUARDAR GRÁFICOS 3D TEMPORALMENTE ---
                img_path_1 = "temp_plot1.png"
                img_path_2 = "temp_plot2.png"
                fig_prev.savefig(img_path_1, bbox_inches='tight', dpi=150)
                plt.close(fig_prev)
                fig_gdl.savefig(img_path_2, bbox_inches='tight', dpi=150)
                plt.close(fig_gdl)

                # --- ENCABEZADO INSTITUCIONAL ---
                img_left = "logo_uns.jpg" if os.path.exists("logo_uns.jpg") else None
                img_right = "logo_ingenieria.png" if os.path.exists("logo_ingenieria.png") else None

                col_left = RLImage(img_left, width=45, height=45) if img_left else Paragraph("<b>UNS</b>", styles['Normal'])
                col_right = RLImage(img_right, width=45, height=45) if img_right else Paragraph("<b>3D</b>", styles['Normal'])
                
                title_style = ParagraphStyle(
                    'HeaderTitle',
                    parent=styles['Heading2'],
                    alignment=1,
                    fontSize=10,
                    leading=13
                )
                
                center_text = Paragraph(
                    "<b>UNIVERSIDAD NACIONAL DEL SANTA</b><br/>FACULTAD DE INGENIERÍA - ESCUELA PROFESIONAL DE INGENIERÍA CIVIL<br/><font color='red'><b>Creado y desarrollado por Yelver Siguas Villanueva</b></font>",
                    title_style
                )

                header_table = Table([[col_left, center_text, col_right]], colWidths=[50, 430, 50])
                header_table.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 10),
                ]))
                
                story.append(header_table)
                story.append(Spacer(1, 10))
                story.append(Paragraph("MEMORIA DE CÁLCULO: ANÁLISIS DE ARMADURA ESPACIAL 3D", styles['Heading1']))
                story.append(Spacer(1, 10))

                # --- INCRUSTAR VISORES 3D ---
                story.append(Paragraph("<b>Visualización Gráfica de la Estructura y Asignación GDL:</b>", styles['Heading2']))
                story.append(Spacer(1, 5))
                img_table_data = [[RLImage(img_path_1, width=220, height=170), RLImage(img_path_2, width=220, height=170)]]
                img_table = Table(img_table_data, colWidths=[250, 250])
                img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
                story.append(img_table)
                story.append(Spacer(1, 15))

                # --- PROCEDIMIENTO DETALLADO EN MATRICES CON CORCHETES ---
                story.append(Paragraph("<b>1. Procedimiento Detallado y Memoria Descriptiva:</b>", styles['Heading2']))
                story.append(Spacer(1, 5))
                
                story.append(Paragraph("<b>Cosenos Directores por Elemento:</b>", styles['Heading3']))
                for m in matrices_elementos:
                    story.append(Paragraph(f"• Barra {m['id']} (N{m['ini']} -> N{m['fin']}): L = {m['L']:.3f} m | λx = {m['cx']:.4f}, λy = {m['cy']:.4f}, λz = {m['cz']:.4f}", styles['Normal']))
                story.append(Spacer(1, 8))

                story.append(Paragraph("<b>Matriz Global del Sistema (K_global):</b>", styles['Heading3']))
                story.append(crear_tabla_matriz_con_corchetes(K_global, decimales=1))
                story.append(Spacer(1, 12))

                story.append(Paragraph("<b>Matriz Particionada de Rigidez Libre (K_LL):</b>", styles['Heading3']))
                story.append(crear_tabla_matriz_con_corchetes(K_LL, decimales=1))
                story.append(Spacer(1, 12))

                story.append(Paragraph("<b>Vector de Desplazamientos Nodales Libres (D_L):</b>", styles['Heading3']))
                story.append(crear_tabla_matriz_con_corchetes(D_libres, decimales=6))
                story.append(Spacer(1, 12))

                story.append(Paragraph("<b>Vector de Reacciones en Apoyos (R_R):</b>", styles['Heading3']))
                story.append(crear_tabla_matriz_con_corchetes(F_RR, decimales=2))
                story.append(Spacer(1, 15))

                # --- RESULTADOS FINALES EN EL PDF ---
                story.append(Paragraph("<b>2. Resultados Finales del Análisis:</b>", styles['Heading2']))
                story.append(Spacer(1, 5))
                story.append(Paragraph("<b>Desplazamientos Nodales:</b>", styles['Heading3']))
                for nid in nodo_ids:
                    base = nodo_a_gdl_indices[nid]
                    story.append(Paragraph(f"• Nodo {nid} -> dX: {D_global[base[0]]:.5f} m | dY: {D_global[base[1]]:.5f} m | dZ: {D_global[base[2]]:.5f} m", styles['Normal']))

                story.append(Spacer(1, 5))
                story.append(Paragraph("<b>Reacciones en Apoyos:</b>", styles['Heading3']))
                for nid in nodo_ids:
                    base = nodo_a_gdl_indices[nid]
                    rx, ry, rz = R_global[base[0]], R_global[base[1]], R_global[base[2]]
                    if any(restricciones[nid]):
                        story.append(Paragraph(f"• Apoyo Nodo {nid} -> Rx: {rx:.2f} KN | Ry: {ry:.2f} KN | Rz: {rz:.2f} KN", styles['Normal']))

                story.append(Spacer(1, 5))
                story.append(Paragraph("<b>Fuerzas Axiales en Barras:</b>", styles['Heading3']))
                for bid, data in resultados_fuerzas.items():
                    story.append(Paragraph(f"• Barra {bid} (N{data['ini']} -> N{data['fin']}): {data['fval']:.2f} KN [{data['estado']}]", styles['Normal']))

                doc.build(story)
                
                # Limpieza de archivos temporales
                if os.path.exists(img_path_1): os.remove(img_path_1)
                if os.path.exists(img_path_2): os.remove(img_path_2)

                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Descargar PDF Oficial con Matrices en Corchetes",
                        data=pdf_file,
                        file_name="Memoria_Calculo_Armadura_3D.pdf",
                        mime="application/pdf"
                    )
except Exception as e:
    st.error(f"Error en la configuración geométrica o de apoyos: {e}")