import streamlit as st
import numpy as np
from scipy.optimize import linprog

def simplex_solver(c, A_ub, b_ub, A_eq, b_eq, var_names, change_desired=None):
    # Resolver o problema de PPL usando scipy.optimize.linprog
    result = linprog(c=-np.array(c), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")

    # Obter resultados
    optimal_point = result.x
    optimal_value = -result.fun
    shadow_prices = result.slack if A_ub else [0] * len(b_ub)

    # Verificar alterações desejadas
    if change_desired:
        modified_b_ub = [b_ub[i] + change_desired[i] for i in range(len(b_ub))]
        modified_result = linprog(c=-np.array(c), A_ub=A_ub, b_ub=modified_b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")
        is_feasible = modified_result.success
        new_optimal_value = -modified_result.fun if is_feasible else None
    else:
        is_feasible = None
        new_optimal_value = None

    return {
        "optimal_point": optimal_point,
        "optimal_value": optimal_value,
        "shadow_prices": shadow_prices,
        "is_feasible": is_feasible,
        "new_optimal_value": new_optimal_value,
    }

# Interface Streamlit
st.title("Solver de PPL com Simplex Tableau")
st.markdown("**Resolva problemas de Programação Linear com até 4 variáveis.**")

# Entradas do usuário
st.header("Entradas do Problema")

n_vars = st.selectbox("Número de variáveis (2, 3 ou 4):", [2, 3, 4])

st.subheader("Coeficientes da Função Objetivo")
c = [st.number_input(f"Coeficiente de x{i+1}:", value=0.0, key=f"c_{i}") for i in range(n_vars)]

st.subheader("Coeficientes das Restrições")
n_restrictions = st.number_input("Número de restrições:", min_value=1, max_value=5, step=1)

A_ub = []
b_ub = []
A_eq = []
b_eq = []

for i in range(int(n_restrictions)):
    st.markdown(f"### Restrição {i+1}", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3, 1, 2])

    with col1:
        coef = [st.number_input(f"Coeficiente de x{j+1} para restrição {i+1}:", value=0.0, key=f"a_{i}_{j}") for j in range(n_vars)]

    with col2:
        constraint_type = st.selectbox(
            "Tipo:",
            options=["<=", "=", ">="],
            index=0,
            key=f"constraint_{i}"
        )

    with col3:
        value = st.number_input(f"Valor máximo da restrição {i+1}:", value=0.0, key=f"b_{i}")

    if constraint_type == "<=":
        A_ub.append(coef)
        b_ub.append(value)
    elif constraint_type == "=":
        A_eq.append(coef)
        b_eq.append(value)
    elif constraint_type == ">=":
        A_ub.append([-x for x in coef])
        b_ub.append(-value)

st.subheader("Alterações Desejadas nas Restrições (Opcional)")
change_desired = [st.number_input(f"Alteração na restrição {i+1}:", value=0.0, key=f"change_{i}") for i in range(len(b_ub))]

if st.button("Resolver PPL"):
    st.header("Resultados")
    results = simplex_solver(
        c, 
        A_ub=np.array(A_ub) if A_ub else None, 
        b_ub=np.array(b_ub) if b_ub else None, 
        A_eq=np.array(A_eq) if A_eq else None, 
        b_eq=np.array(b_eq) if b_eq else None,
        var_names=[f"x{i+1}" for i in range(n_vars)], 
        change_desired=change_desired
    )

    st.subheader("Ponto Ótimo de Operação")
    for i, val in enumerate(results["optimal_point"]):
        st.write(f"x{i+1} = {val:.2f}")
    st.write(f"Lucro Ótimo: {results['optimal_value']:.2f}")

    st.subheader("Preço-Sombra (Slack)")
    for i, val in enumerate(results["shadow_prices"]):
        st.write(f"Restrição {i+1}: {val:.2f}")

    st.subheader("Alterações Desejadas")
    if results["is_feasible"] is not None:
        if results["is_feasible"]:
            st.success("As alterações são viáveis.")
            st.write(f"Novo Lucro Ótimo: {results['new_optimal_value']:.2f}")
        else:
            st.error("As alterações não são viáveis.")
