import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

st.set_page_config(
    page_title="Simplex Solver Pro",
    page_icon="🔢",
    layout="wide"
)

if "resolvido" not in st.session_state:
    st.session_state.resolvido = False
    st.session_state.restrictions_info = []
    st.session_state.c = []
    st.session_state.A_ub = []
    st.session_state.b_ub = []
    st.session_state.A_eq = []
    st.session_state.b_eq = []
    st.session_state.res = None
    st.session_state.num_vars = 2

with st.sidebar:
    st.header("⚙️ Configurações do Problema")
    num_vars = st.selectbox(
        "Quantidade de variáveis",
        [2, 3, 4],
        key="num_vars_select",
        help="Escolha o número de variáveis de decisão do seu problema."
    )
    num_restr = st.number_input(
        "Quantidade de restrições",
        min_value=1,
        max_value=15,
        step=1,
        key="num_restr_input",
        help="Quantas restrições o seu problema possui?"
    )

    st.info("Após resolver, os resultados aparecerão na tela principal em abas.")
    
    if st.button("🔁 Resetar Aplicação", use_container_width=True):
        # Limpa todo o estado da sessão e reinicia
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

st.title("🔢 Simplex Solver Pro")
st.markdown("""
Esta aplicação resolve Problemas de Programação Linear (PPL) usando o método Simplex.
**Recursos:**
- **Função Objetivo:** Defina os coeficientes para maximização.
- **Restrições:** Adicione restrições do tipo ≤, ≥ ou =.
- **Resultados Claros:** Visualize a solução ótima e o valor da função objetivo de forma destacada.
- **Análise de Sensibilidade:** Entenda o impacto de cada restrição através do preço-sombra e teste variações nos recursos.
- **Gráfico Interativo (2D):** Para problemas com 2 variáveis, explore a região viável e a solução ótima graficamente.
""")

st.header("📝 Definição do Problema")

st.subheader("Função Objetivo (Maximizar Z)")
with st.container(border=True):
    cols_c = st.columns(num_vars)
    c_input = []
    for i in range(num_vars):
        c_input.append(cols_c[i].number_input(f"Coef. de x{i+1}", key=f"c_coef_{i}", format="%.2f"))

st.subheader("Restrições")
A_ub, b_ub, A_eq, b_eq = [], [], [], []
restrictions_info = []

for i in range(int(num_restr)):
    with st.container(border=True):
        st.markdown(f"**Restrição {i+1}**")
        cols_r = st.columns(num_vars + 2)
        row = []
        for j in range(num_vars):
            coef = cols_r[j].number_input(f"x{j+1}", key=f"A_coef_{i}_{j}", format="%.2f", label_visibility="collapsed")
            row.append(coef)
        
        operador = cols_r[-2].selectbox("Operador", ["<=", ">=", "="], key=f"op_{i}", label_visibility="collapsed")
        bi = cols_r[-1].number_input(f"b{i+1}", key=f"b_val_{i}", format="%.2f", label_visibility="collapsed")
        
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

if st.button("🚀 Resolver Problema", type="primary", use_container_width=True):
    with st.spinner("Calculando a solução ótima..."):
        try:
            c = [-val for val in c_input]

            res = linprog(c,
                          A_ub=A_ub if A_ub else None, b_ub=b_ub if b_ub else None,
                          A_eq=A_eq if A_eq else None, b_eq=b_eq if b_eq else None,
                          method='highs')

            if res.success:
                st.session_state.resolvido = True
                st.session_state.restrictions_info = restrictions_info
                st.session_state.c_input = c_input
                st.session_state.c = c
                st.session_state.A_ub = A_ub
                st.session_state.b_ub = b_ub
                st.session_state.A_eq = A_eq
                st.session_state.b_eq = b_eq
                st.session_state.res = res
                st.session_state.num_vars = num_vars
                st.success("✅ Solução ótima encontrada com sucesso!")
            else:
                st.error(f"❌ Não foi possível encontrar uma solução. Mensagem do Solver: **{res.message}**")
                st.session_state.resolvido = False

        except ValueError as e:
            st.error(f"❌ Erro nos dados de entrada. Verifique os valores fornecidos. Detalhes: {e}")
            st.session_state.resolvido = False

if st.session_state.resolvido:
    res = st.session_state.res
    num_vars = st.session_state.num_vars
    
    st.header("📊 Resultados da Otimização")

    tab1, tab2, tab3 = st.tabs(["📝 **Resumo da Solução**", "📈 **Análise Gráfica (2D)**", "🔬 **Análise de Sensibilidade**"])

    with tab1:
        st.subheader("Solução Ótima")
        
        cols_metrics = st.columns(num_vars + 1)
        cols_metrics[0].metric(label="Valor Máximo de Z", value=f"{-res.fun:.4f}")
        for i, val in enumerate(res.x):
            cols_metrics[i+1].metric(label=f"Valor de x{i+1}", value=f"{val:.4f}")

        st.subheader("Análise das Restrições (Preço-Sombra)")
        
        shadow_prices = []
        slacks = []
        status = []
        
        ineq_idx = 0
        eq_idx = 0

        for r_info in st.session_state.restrictions_info:
            op = r_info['operator']
            valor_lado_esquerdo = np.dot(r_info['row'], res.x)
            
            if op == "<=":
                price = res.ineqlin.marginals[ineq_idx] if hasattr(res, 'ineqlin') and ineq_idx < len(res.ineqlin.marginals) else 0
                slack = r_info['bi'] - valor_lado_esquerdo
                slacks.append(f"{slack:.4f} (Folga)")
                shadow_prices.append(f"{-price:.4f}")
                status.append("Ativa" if np.isclose(slack, 0) else "Inativa")
                ineq_idx += 1
            elif op == ">=":
                price = res.ineqlin.marginals[ineq_idx] if hasattr(res, 'ineqlin') and ineq_idx < len(res.ineqlin.marginals) else 0
                slack = valor_lado_esquerdo - r_info['bi']
                slacks.append(f"{slack:.4f} (Excesso)")
                shadow_prices.append(f"{price:.4f}")
                status.append("Ativa" if np.isclose(slack, 0) else "Inativa")
                ineq_idx += 1
            else:
                price = res.eqlin.marginals[eq_idx] if hasattr(res, 'eqlin') and eq_idx < len(res.eqlin.marginals) else 0
                slacks.append("N/A (Igualdade)")
                shadow_prices.append(f"{-price:.4f}")
                status.append("Ativa")
                eq_idx += 1
        
        df_shadow = pd.DataFrame({
            'Restrição': [f"Restrição {i+1}" for i in range(len(st.session_state.restrictions_info))],
            'Status': status,
            'Folga/Excesso': slacks,
            'Preço-Sombra': shadow_prices
        })
        st.dataframe(df_shadow, use_container_width=True)
        st.info("O **Preço-Sombra** indica o quanto o valor da função objetivo (Z) aumentaria se o lado direito (recurso) de uma restrição ativa fosse aumentado em uma unidade.")

    with tab2:
        if num_vars == 2:
            st.subheader("Visualização Gráfica da Solução")
            
            x1_vals = np.linspace(0, max(res.x[0] * 2, 20), 400)
            x2_vals = np.linspace(0, max(res.x[1] * 2, 20), 400)
            x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
            
            feasible_region = np.ones_like(x1_grid, dtype=bool)

            fig, ax = plt.subplots(figsize=(10, 8))

            for r_info in st.session_state.restrictions_info:
                row, bi, op = r_info['row'], r_info['bi'], r_info['operator']
                expr = row[0] * x1_grid + row[1] * x2_grid
                if op == "<=":
                    feasible_region &= (expr <= bi)
                elif op == ">=":
                    feasible_region &= (expr >= bi)
                else:
                    feasible_region &= np.isclose(expr, bi)

            ax.imshow(
                feasible_region.astype(int),
                origin='lower',
                extent=(x1_vals.min(), x1_vals.max(), x2_vals.min(), x2_vals.max()),
                cmap="Greens",
                alpha=0.3,
                aspect='auto'
            )

            for r_info in st.session_state.restrictions_info:
                row, bi, op = r_info['row'], r_info['bi'], r_info['operator']
                label = f"{row[0]:.2f}x1 + {row[1]:.2f}x2 {op} {bi:.2f}"
                if np.abs(row[1]) > 1e-6:
                    y = (bi - row[0] * x1_vals) / row[1]
                    ax.plot(x1_vals, y, label=label)
                elif np.abs(row[0]) > 1e-6:
                    ax.axvline(x=bi/row[0], label=label, linestyle='--')
            
            ax.plot(res.x[0], res.x[1], 'ro', markersize=10, label=f'Ponto Ótimo ({res.x[0]:.2f}, {res.x[1]:.2f})')

            ax.set_xlim(0, x1_vals.max())
            ax.set_ylim(0, x2_vals.max())
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        else:
            st.info("A visualização gráfica só está disponível para problemas com 2 variáveis.")

    with tab3:
        st.subheader("Teste de Variações nos Lados Direitos (b)")
        st.info("Altere o valor do lado direito (RHS ou 'b') de cada restrição para simular mudanças nos recursos disponíveis e ver o impacto na solução ótima.")

        deltas = []
        cols_b = st.columns(len(st.session_state.restrictions_info))
        for i, r_info in enumerate(st.session_state.restrictions_info):
            with cols_b[i]:
                with st.container(border=True):
                    st.markdown(f"**Restrição {i+1}**")
                    original_bi = r_info['bi']
                    new_bi = st.number_input(f"Novo valor de b (Original: {original_bi})", value=original_bi, format="%.2f", key=f"new_b_{i}")
                    deltas.append(new_bi - original_bi)
        
        if st.button("🧪 Testar Novas Variações", use_container_width=True):
            with st.spinner("Recalculando com as novas variações..."):
                b_ub_temp, b_eq_temp = [], []
                A_ub_temp, A_eq_temp = [], []

                for i, r_info in enumerate(st.session_state.restrictions_info):
                    b_mod = r_info['bi'] + deltas[i]
                    if r_info['operator'] == '<=':
                        A_ub_temp.append(r_info['row'])
                        b_ub_temp.append(b_mod)
                    elif r_info['operator'] == '>=':
                        A_ub_temp.append([-a for a in r_info['row']])
                        b_ub_temp.append(-b_mod)
                    else:
                        A_eq_temp.append(r_info['row'])
                        b_eq_temp.append(b_mod)
                
                try:
                    res_test = linprog(st.session_state.c,
                                       A_ub=A_ub_temp if A_ub_temp else None, b_ub=b_ub_temp if b_ub_temp else None,
                                       A_eq=A_eq_temp if A_eq_temp else None, b_eq=b_eq_temp if b_eq_temp else None,
                                       method='highs')
                    
                    st.subheader("Resultado do Teste")
                    if res_test.success:
                        st.success("✅ As variações propostas mantêm o problema com uma solução viável!")
                        col1, col2 = st.columns(2)
                        col1.metric("Novo Valor Ótimo de Z", f"{-res_test.fun:.4f}")
                        col2.metric("Variação em Z", f"{-res_test.fun - (-res.fun):.4f}")
                        
                        for i, val in enumerate(res_test.x):
                            st.write(f"**Novo x{i+1}** = {val:.4f}")
                    else:
                        st.error(f"❌ As variações tornaram o problema inviável ou ilimitado. Mensagem: **{res_test.message}**")
                except ValueError as e:
                    st.error(f"❌ Erro ao testar variações: {e}")
