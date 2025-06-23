import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

st.set_page_config(page_title="Simplex - Programação Linear", layout="centered")
st.title("🔢 Resolução de PPL com Simplex (até 4 variáveis)")


st.markdown("""
Este aplicativo resolve problemas de Programação Linear (PPL) usando o **método Simplex**, com até 4 variáveis.
Você pode:
- Inserir a função objetivo e restrições
- Analisar o **preço-sombra** e a solução ótima
- Alterar os valores dos lados direitos (b) de todas as restrições ao mesmo tempo
- Testar se essas variações ainda produzem uma solução viável
""")

if "resolvido" not in st.session_state:
    st.session_state.resolvido = False
    st.session_state.restrictions_info = []

# Entrada do problema
num_vars = st.selectbox("Quantidade de variáveis", [2, 3, 4], key="num_vars_select")
st.subheader("Função Objetivo (Maximização)")
c = [-st.number_input(f"Coeficiente de x{i+1}", key=f"c_coef_{i}") for i in range(num_vars)]

num_restr = st.number_input("Quantidade de restrições", min_value=1, max_value=10, step=1, key="num_restr_input")
A_ub, b_ub, A_eq, b_eq = [], [], [], []
restrictions_info = [] 

st.subheader("Restrições")
for i in range(int(num_restr)):
    row = []
    st.markdown(f"**Restrição {i+1}**")
    cols = st.columns(num_vars + 2)
    for j in range(num_vars):
        coef = cols[j].number_input(f"x{j+1}", key=f"A_coef_{i}_{j}", format="%.2f")
        row.append(coef)
    operador = cols[-2].selectbox("Operador", ["<=", ">=", "="], key=f"op_{i}")
    bi = cols[-1].number_input(f"b{i+1}", key=f"b_val_{i}", format="%.2f")
    restrictions_info.append({"row": row, "bi": bi, "operator": operador, "original_index": i})

    if operador == "<=":
        A_ub.append(row)
        b_ub.append(bi)
    elif operador == ">=":
        A_ub.append([-a for a in row])
        b_ub.append(-bi)
    else: 
        A_eq.append(row)
        b_eq.append(bi)

# Botão para resolver o PPL
if st.button("Resolver PPL", key="solve_button"):
    try:
        res = linprog(c, A_ub=A_ub if A_ub else None, b_ub=b_ub if b_ub else None,
                      A_eq=A_eq if A_eq else None, b_eq=b_eq if b_eq else None, method='highs')

        if res.success:
            st.session_state.resolvido = True
            st.session_state.restrictions_info = restrictions_info
            st.session_state.c = c
            st.session_state.A_ub = A_ub
            st.session_state.b_ub = b_ub
            st.session_state.A_eq = A_eq
            st.session_state.b_eq = b_eq
            st.session_state.res = res
            st.session_state.num_vars = num_vars 

            st.success("✅ Solução encontrada!")
        else:
            st.error(f"❌ Problema inviável ou sem solução ótima. Mensagem: {res.message}")
            st.session_state.resolvido = False
    except ValueError as e:
        st.error(f"❌ Erro na entrada dos dados ou na resolução: {e}")
        st.session_state.resolvido = False

if st.session_state.resolvido:
    res = st.session_state.res
    num_vars = st.session_state.num_vars

    st.subheader("🔍 Resultado Ótimo")
    for i, val in enumerate(res.x):
        st.write(f"x{i+1} = {val:.2f}")
    st.write(f"Valor ótimo da função objetivo: **{-res.fun:.2f}**")

   # Preço-sombra
    st.subheader("💸 Preços-Sombra das Restrições:")
    ineq_marginal_idx = 0
    eq_marginal_idx = 0

    for i, r_info in enumerate(st.session_state.restrictions_info):
        original_operator = r_info['operator']

        if original_operator == "<=":
            if hasattr(res, 'ineqlin') and ineq_marginal_idx < len(res.ineqlin.marginals):
                price = res.ineqlin.marginals[ineq_marginal_idx]  # sem inverter
                st.write(f"Restrição {r_info['original_index']+1} (Original: {original_operator}): {price:.4f}")
                ineq_marginal_idx += 1
            else:
                st.write(f"Restrição {r_info['original_index']+1} (Original: {original_operator}): N/A")
        elif original_operator == ">=":
            if hasattr(res, 'ineqlin') and ineq_marginal_idx < len(res.ineqlin.marginals):
                price = res.ineqlin.marginals[ineq_marginal_idx]  # também sem inverter
                st.write(f"Restrição {r_info['original_index']+1} (Original: {original_operator}): {price:.4f}")
                ineq_marginal_idx += 1
            else:
                st.write(f"Restrição {r_info['original_index']+1} (Original: {original_operator}): N/A")
        elif original_operator == "=":
            if hasattr(res, 'eqlin') and eq_marginal_idx < len(res.eqlin.marginals):
                price = -res.eqlin.marginals[eq_marginal_idx]  # apenas aqui inverte
                st.write(f"Restrição {r_info['original_index']+1} (Original: {original_operator}): {price:.4f}")
                eq_marginal_idx += 1
            else:
                st.write(f"Restrição {r_info['original_index']+1} (Original: {original_operator}): N/A")

    st.markdown("---")
   

    # Gráfico (apenas para 2 variáveis)
    if num_vars == 2:
        st.subheader("📈 Visualização Gráfica")

        plt.figure(figsize=(8, 6))

        x1_max = max(res.x[0] * 1.5, 10.0)
        y_max_calc = 0.0
        for r_info in st.session_state.restrictions_info:
            row_coeffs = r_info['row']
            bi_val = r_info['bi']
            if len(row_coeffs) > 1 and row_coeffs[1] != 0:
                y_at_x1_0 = bi_val / row_coeffs[1]
                if y_at_x1_0 > y_max_calc:
                    y_max_calc = y_at_x1_0
            elif row_coeffs[0] != 0:
                x_at_y1_0 = bi_val / row_coeffs[0]
                if x_at_y1_0 > x1_max:
                    x1_max = x_at_y1_0 * 1.2

        y_max = max(y_max_calc * 1.2, res.x[1] * 1.5, 10.0) 
        x1_plot = np.linspace(0, x1_max, 400)

        for idx, r_info in enumerate(st.session_state.restrictions_info):
            row_coeffs = r_info['row']
            bi_val = r_info['bi']
            op = r_info['operator']

            if len(row_coeffs) > 1 and row_coeffs[1] != 0:
                y = (bi_val - row_coeffs[0]*x1_plot) / row_coeffs[1]
                plt.plot(x1_plot, y, label=f"Restrição {r_info['original_index']+1} ({op})")
            elif row_coeffs[0] != 0:
                x_const = bi_val / row_coeffs[0]
                plt.axvline(x=x_const, label=f"Restrição {r_info['original_index']+1} ({op})", linestyle='--')

        # Plota o ponto ótimo
        plt.plot(res.x[0], res.x[1], 'ro', markersize=8, label='Ponto Ótimo')

        plt.axhline(0, color='black') # Eixo x
        plt.axvline(0, color='black') # Eixo y
        plt.xlim(left=0, right=x1_max)
        plt.ylim(bottom=0, top=y_max)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

    st.header("📊 Análise de Variações nos Lados Direitos (b)")
   

    deltas = []
    for i in range(len(st.session_state.restrictions_info)):
        delta = st.number_input(f"Δ b da restrição {i+1}", value=0.0, format="%.2f", key=f"delta_b_input_{i}")
        deltas.append(delta)

    if st.button("Testar variações", key="test_variations_button"):
        A_ub_temp, b_ub_temp, A_eq_temp, b_eq_temp = [], [], [], []
        for i, r_info in enumerate(st.session_state.restrictions_info):
            original_row = r_info['row']
            original_bi = r_info['bi']
            original_operator = r_info['operator']

            b_mod = original_bi + deltas[i]

            if original_operator == "<=":
                A_ub_temp.append(original_row)
                b_ub_temp.append(b_mod)
            elif original_operator == ">=":
                A_ub_temp.append([-a for a in original_row])
                b_ub_temp.append(-b_mod)
            else:
                A_eq_temp.append(original_row)
                b_eq_temp.append(b_mod)

        try:
            res_test = linprog(st.session_state.c,
                                A_ub=A_ub_temp if A_ub_temp else None, b_ub=b_ub_temp if b_ub_temp else None,
                                A_eq=A_eq_temp if A_eq_temp else None, b_eq=b_eq_temp if b_eq_temp else None,
                                method='highs')

            st.subheader("🧪 Resultado após as variações:")
            if res_test.success:
                st.success("✅ Variações válidas! Problema ainda possui solução.")
                st.write(f"Novo valor ótimo da função objetivo: **{-res_test.fun:.2f}**")
                for i, val in enumerate(res_test.x):
                    st.write(f"x{i+1} = {val:.2f}")
            else:
                st.error(f"❌ As variações tornam o problema inviável ou sem solução ótima. Mensagem: {res_test.message}")
        except ValueError as e:
            st.error(f"❌ Erro ao testar variações: {e}")

# Botão de reset
if st.button("🔁 Resetar", key="reset_button"):
    st.session_state.clear() 
    st.rerun() 
