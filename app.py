from flask import Flask, request, jsonify
from flask_cors import CORS
from modelo_previsao import (
    carregar_e_preparar_dados,
    treinar_modelo_e_prever,
    simular_estoque_e_gerar_dashboard,
    calcular_metricas
)

app = Flask(__name__)
CORS(app)

@app.route('/api/prever', methods=['POST'])
def prever():
    try:
        # Parâmetros recebidos
        dados = request.get_json()
        id_produto = int(dados.get('id_produto', 237497))
        caminho_csv = dados.get('arquivo_csv', 'dados_vendas_sinteticos.csv')
        colunas_exogenas = ['promocao', 'feriado']

        # Execução do pipeline
        treino_y, teste_y, treino_X, teste_X, serie_completa, exogenas_df = carregar_e_preparar_dados(
            caminho_csv, id_produto, colunas_exogenas
        )

        dias_para_prever = len(teste_y) + 2
        exogenas_futuras = None

        if exogenas_df is not None:
            datas_futuras = pd.date_range(start=exogenas_df.index[-1] + pd.Timedelta(days=1), periods=dias_para_prever, freq='D')
            exogenas_futuras = pd.DataFrame(0, index=datas_futuras, columns=colunas_exogenas)
            exog_para_previsao = pd.concat([teste_X, exogenas_futuras])
        else:
            exog_para_previsao = None

        resultado_modelo, previsao_df = treinar_modelo_e_prever(treino_y, treino_X, dias_para_prever, exog_para_previsao)
        nome_dashboard = simular_estoque_e_gerar_dashboard(treino_y, teste_y, serie_completa, previsao_df, id_produto, resultado_modelo)

        # Métricas
        previsao_teste = previsao_df.head(len(teste_y))['mean']
        mape, rmse = calcular_metricas(teste_y, previsao_teste)

        # Quantidades (D+1 e D+2)
        perda_peso = 0.15
        demanda_d1 = previsao_df['mean'].iloc[-2]
        demanda_d2 = previsao_df['mean'].iloc[-1]
        qtd_retirar_hoje = demanda_d2 / (1 - perda_peso)
        qtd_em_descongelamento = demanda_d1 / (1 - perda_peso)
        qtd_disponivel_hoje = treino_y.iloc[-1] / (1 - perda_peso)

        return jsonify({
            "status": "sucesso",
            "dashboard": nome_dashboard,
            "mape": mape,
            "rmse": rmse,
            "qtd_retirar_hoje": round(qtd_retirar_hoje, 2),
            "qtd_em_descongelamento": round(qtd_em_descongelamento, 2),
            "qtd_disponivel_hoje": round(qtd_disponivel_hoje, 2)
        })

    except Exception as e:
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
