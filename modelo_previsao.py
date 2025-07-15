"""
Este script realiza uma análise de séries temporais de dados de vendas para prever a demanda futura
de um produto específico. Ele utiliza um modelo SARIMAX para a previsão e, com base nela,
simula uma lógica de negócio para otimização do processo de descongelamento de produtos,
visando garantir que a quantidade correta de produto esteja disponível para venda a cada dia,
minimizando perdas e rupturas.

O pipeline do script consiste em:
1.  **Carregamento e Preparação dos Dados**: Lê um arquivo CSV de vendas, filtra os dados para
    um SKU (ID de produto) específico, trata a série temporal para garantir frequência diária
    e a divide em conjuntos de treinamento e teste.
2.  **Treinamento do Modelo**: Treina um modelo SARIMAX com os dados históricos de vendas.
    Os parâmetros do modelo são pré-definidos para capturar tendências e sazonalidades (neste caso, semanais).
3.  **Geração de Previsões**: Utiliza o modelo treinado para prever a demanda de vendas para os
    próximos dias.
4.  **Simulação de Estoque e Geração de Dashboard**: Com base nas previsões, aplica uma lógica
    de negócio que calcula a quantidade de produto a ser retirada do freezer diariamente.
    As informações, incluindo a previsão, métricas de erro do modelo (MAPE e RMSE) e o
    plano de descongelamento, são compiladas e exibidas em um dashboard interativo
    gerado com a biblioteca Plotly.

O resultado final é um arquivo HTML que contém o dashboard, fornecendo uma visão clara
e acionável para a equipe de operações.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def carregar_e_preparar_dados(caminho_arquivo, id_produto, colunas_exogenas=None):
    """
    Carrega os dados de vendas de um arquivo CSV, filtra por um produto específico,
    prepara a série temporal e as variáveis exógenas, e divide em treino e teste.

    Args:
        caminho_arquivo (str): O caminho para o arquivo CSV.
        id_produto (int): O ID do produto a ser analisado.
        colunas_exogenas (list, optional): Lista de nomes das colunas a serem usadas como
                                           variáveis exógenas. Defaults to None.

    Returns:
        tuple: Uma tupla contendo:
               - pd.Series: Série temporal de vendas para treino.
               - pd.Series: Série temporal de vendas para teste.
               - pd.DataFrame: Variáveis exógenas para treino.
               - pd.DataFrame: Variáveis exógenas para teste.
               - pd.Series: A série temporal completa de vendas.
               - pd.DataFrame: As variáveis exógenas completas.
    """
    dados = pd.read_csv(caminho_arquivo)
    dados['data_dia'] = pd.to_datetime(dados['data_dia'])
    dados_produto = dados[dados['id_produto'] == id_produto].copy()
    dados_produto.set_index('data_dia', inplace=True)
    
    # Assegura frequência diária, preenchendo dados ausentes.
    dados_completos = dados_produto.asfreq('D')
    serie_temporal = dados_completos['total_venda_dia_kg'].fillna(0)
    
    exogenas_df = None
    exogenas_df = None
    if colunas_exogenas:
        # Verifica quais das colunas exógenas solicitadas realmente existem no DataFrame
        colunas_existentes = [col for col in colunas_exogenas if col in dados_produto.columns]
        
        if colunas_existentes:
            exogenas_df = dados_completos[colunas_existentes].fillna(0)
            # Garante que colunas não encontradas não causem problemas futuros
            colunas_exogenas = colunas_existentes
        else:
            print(f"Aviso: Nenhuma das colunas exógenas especificadas {colunas_exogenas} foi encontrada no arquivo. O modelo será treinado sem variáveis externas.")
            colunas_exogenas = []

    # Dividir em treino e teste (últimos 7 dias para teste)
    treino_y = serie_temporal[:-7]
    teste_y = serie_temporal[-7:]
    
    treino_X = None
    teste_X = None
    if exogenas_df is not None:
        treino_X = exogenas_df[:-7]
        teste_X = exogenas_df[-7:]

    return treino_y, teste_y, treino_X, teste_X, serie_temporal, exogenas_df

def treinar_modelo_e_prever(treino_data, exog_data, steps, exog_futuro=None):
    """
    Treina um modelo SARIMAX e o utiliza para fazer previsões futuras, incluindo variáveis exógenas.

    Args:
        treino_data (pd.Series): A série temporal de dados de treinamento.
        exog_data (pd.DataFrame): DataFrame com as variáveis exógenas para o período de treino.
        steps (int): O número de passos (dias) à frente para prever.
        exog_futuro (pd.DataFrame, optional): DataFrame com os valores futuros das variáveis exógenas
                                             para o período de previsão. Defaults to None.

    Returns:
        tuple: Uma tupla contendo:
               - SARIMAXResults: O objeto com os resultados do modelo treinado.
               - pd.DataFrame: Um DataFrame com as previsões, incluindo média e intervalos de confiança.
    """
    modelo = SARIMAX(treino_data, 
                     exog=exog_data,
                     order=(1, 1, 1), 
                     seasonal_order=(1, 1, 1, 7),
                     enforce_stationarity=False, enforce_invertibility=False)
    resultado = modelo.fit(disp=False)
    previsao = resultado.get_forecast(steps=steps, exog=exog_futuro)
    return resultado, previsao.summary_frame(alpha=0.05)

def calcular_metricas(real, previsto):
    """
    Calcula e formata as métricas de erro MAPE (Mean Absolute Percentage Error) e RMSE (Root Mean Squared Error).

    Args:
        real (array-like): Os valores reais (verdadeiros).
        previsto (array-like): Os valores previstos pelo modelo.

    Returns:
        tuple: Uma tupla contendo o MAPE e o RMSE, ambos formatados como strings.
               - str: O MAPE formatado como uma porcentagem com duas casas decimais.
               - str: O RMSE formatado como um número de ponto flutuante com duas casas decimais.
    """
    # Calcula o Erro Percentual Absoluto Médio (MAPE).
    mape = mean_absolute_percentage_error(real, previsto)
    
    # Calcula a Raiz do Erro Quadrático Médio (RMSE).
    rmse = np.sqrt(mean_squared_error(real, previsto))
    
    # Retorna as métricas formatadas como strings para exibição.
    return f'{mape:.2%}', f'{rmse:.2f}'

def simular_estoque_e_gerar_dashboard(treino, teste, serie_completa, previsao_df, id_produto, resultado_modelo=None):
    """
    Simula a lógica de negócio, calcula as quantidades necessárias e gera um dashboard interativo
    com os resultados, previsões e o impacto das variáveis exógenas.

    Args:
        treino (pd.Series): A série temporal de vendas de treinamento.
        teste (pd.Series): A série temporal de vendas de teste.
        serie_completa (pd.Series): A série temporal de vendas completa.
        previsao_df (pd.DataFrame): DataFrame com as previsões do modelo.
        id_produto (int): O ID do produto para o qual o dashboard está sendo gerado.
        resultado_modelo (SARIMAXResults, optional): O objeto com os resultados do modelo treinado
                                                     para exibir os coeficientes. Defaults to None.

    Returns:
        str: O nome do arquivo HTML do dashboard gerado.
    """
    # Define a data atual e a taxa de perda de peso do produto durante o descongelamento.
    data_hoje = pd.to_datetime(datetime.now().date())
    perda_peso = 0.15  # 15% de perda de peso

    # --- Lógica de Negócio para o Relatório ---
    # Extrai as previsões de demanda para os próximos dois dias (D+1 e D+2).
    # iloc[-2] refere-se à previsão para amanhã (D+1).
    # iloc[-1] refere-se à previsão para depois de amanhã (D+2).
    demanda_d1 = previsao_df['mean'].iloc[-2] if len(previsao_df) > 1 else 0
    demanda_d2 = previsao_df['mean'].iloc[-1] if len(previsao_df) > 0 else 0

    # Calcula a quantidade de produto a ser retirada do freezer hoje.
    # Isso é para atender à demanda prevista para D+2, ajustando pela perda de peso.
    qtd_retirar_hoje = demanda_d2 / (1 - perda_peso)
    
    # Calcula a quantidade de produto que já está em processo de descongelamento.
    # Isso foi retirado ontem para atender à demanda de D+1.
    qtd_em_descongelamento = demanda_d1 / (1 - perda_peso)
    
    # Estima a quantidade de produto já descongelado e disponível para venda hoje.
    # Baseia-se na última venda real registrada, ajustada pela perda de peso.
    # Esta é uma simplificação; um sistema real teria dados de inventário.
    qtd_disponivel_hoje = treino.iloc[-1] / (1 - perda_peso)
    
    # Informação textual sobre a idade dos lotes no processo de descongelamento.
    idade_lotes = "1 dia (Estante Central), 2 dias (Estante Direita)"

    # --- Cálculo de Métricas de Performance ---
    # Pega a parte da previsão que corresponde ao período de teste.
    previsao_teste = previsao_df.head(len(teste))['mean']
    # Calcula o MAPE e o RMSE comparando os dados de teste com a previsão para esse período.
    mape, rmse = calcular_metricas(teste, previsao_teste)

    # --- Criação do Dashboard Interativo com Plotly ---
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=('Vendas Históricas vs. Previsão', 
                        'Métricas de Performance', 'Impacto dos Fatores Externos',
                        'Fluxo de Descongelamento', 'Relatório de Descongelamento Diário'),
        specs=[[{"colspan": 2}, None], 
               [{"type": "table"}, {"type": "table"}],
               [{"type": "table", "colspan": 2}, None],
               [{"type": "table", "colspan": 2}, None]],
        vertical_spacing=0.15,
        row_heights=[0.4, 0.15, 0.15, 0.3]
    )

    # --- Adição dos Gráficos e Tabelas ---

    # 1. Gráfico de Linha
    fig.add_trace(go.Scatter(x=serie_completa.index, y=serie_completa, name='Vendas Reais', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=previsao_df.index, y=previsao_df['mean'], name='Previsão de Vendas', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=previsao_df.index, y=previsao_df['mean_ci_upper'], fill='tonexty', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Intervalo de Confiança'), row=1, col=1)
    fig.add_trace(go.Scatter(x=previsao_df.index, y=previsao_df['mean_ci_lower'], fill='tonexty', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)

    # 2. Tabela de Métricas
    fig.add_trace(go.Table(header=dict(values=['<b>Métrica</b>', '<b>Valor</b>'], align='left'), cells=dict(values=[['MAPE', 'RMSE'], [mape, rmse]], align='left')), row=2, col=1)

    # 3. Tabela de Impacto dos Fatores Externos
    if resultado_modelo and hasattr(resultado_modelo, 'summary'):
        coeficientes = resultado_modelo.summary().tables[1]
        coef_df = pd.read_html(coeficientes.as_html(), header=0, index_col=0)[0]
        
        # Filtra para mostrar apenas variáveis exógenas
        exog_coefs = coef_df[coef_df.index.str.startswith('promocao') | coef_df.index.str.startswith('feriado')]

        if not exog_coefs.empty:
            fig.add_trace(go.Table(
                header=dict(values=['<b>Fator Externo</b>', '<b>Coeficiente</b>', '<b>P>|z|</b>'], align='left'),
                cells=dict(values=[exog_coefs.index, exog_coefs['coef'].round(3), exog_coefs['P>|z|'].round(3)], align='left')
            ), row=2, col=2)
        else:
             fig.add_trace(go.Table(header=dict(values=['Fator Externo', 'Impacto']), cells=dict(values=[['Nenhum fator exógeno significativo'], ['N/A']])), row=2, col=2)


    # 4. Tabela do Fluxo de Descongelamento
    fig.add_trace(go.Table(header=dict(values=['<b>Estágio</b>', '<b>Descrição</b>'], align='left'), cells=dict(values=[['Dia 0', 'Dia 1', 'Dia 2'], ['Retirada do Freezer -> Estante Esquerda', 'Move para Estante Central', 'Move para Estante Direita (Pronto para Venda)']], align='left')), row=3, col=1)

    # 5. Tabela Principal do Relatório
    fig.add_trace(go.Table(
        header=dict(values=['<b>Data de Retirada</b>', '<b>SKU</b>', '<b>Qtd. a Retirar Hoje (kg)</b>', '<b>Qtd. em Descongelamento (kg)</b>', '<b>Qtd. Disponível Hoje (kg)</b>', '<b>Idade dos Lotes</b>'], align='left'),
        cells=dict(values=[[data_hoje.strftime('%Y-%m-%d')], [id_produto], [f'{qtd_retirar_hoje:.2f}'], [f'{qtd_em_descongelamento:.2f}'], [f'{qtd_disponivel_hoje:.2f}'], [idade_lotes]], align='left')
    ), row=4, col=1)

    # --- Finalização e Salvamento do Dashboard ---
    fig.update_layout(height=1200, title_text=f"Dashboard de Otimização de Descongelamento - SKU {id_produto}", showlegend=True)
    nome_arquivo = f'dashboard_descongelamento_sku_{id_produto}_{data_hoje.strftime("%Y%m%d")}.html'
    fig.write_html(nome_arquivo)
    return nome_arquivo

if __name__ == "__main__":
    """
    Ponto de entrada do script.
    Define as constantes, orquestra a chamada das funções para carregar dados,
    treinar o modelo, fazer previsões e gerar o dashboard final.
    """
    # --- Configurações ---
    ID_PRODUTO = 237497
    ARQUIVO_VENDAS = 'dados_vendas_sinteticos.csv'
    # Define as colunas que serão usadas como variáveis externas (exógenas).
    # O arquivo CSV de entrada deve conter essas colunas.
    COLUNAS_EXOGENAS = ['promocao', 'feriado']

    # --- Execução do Pipeline ---
    # 1. Carrega e prepara os dados, incluindo as variáveis exógenas.
    treino_y, teste_y, treino_X, teste_X, serie_completa, exogenas_df = \
        carregar_e_preparar_dados(ARQUIVO_VENDAS, ID_PRODUTO, COLUNAS_EXOGENAS)

    # 2. Define o horizonte de previsão e prepara os dados exógenos futuros.
    dias_para_prever = len(teste_y) + 2
    exogenas_futuras = None
    if exogenas_df is not None:
        # Para este exemplo, estamos assumindo que não haverá promoções ou feriados nos próximos dias.
        # Em um cenário real, estes dados viriam de um planejamento de calendário.
        datas_futuras = pd.date_range(start=exogenas_df.index[-1] + pd.Timedelta(days=1), periods=dias_para_prever, freq='D')
        exogenas_futuras = pd.DataFrame(0, index=datas_futuras, columns=COLUNAS_EXOGENAS)
        # Concatena as variáveis exógenas do período de teste com as futuras.
        exog_para_previsao = pd.concat([teste_X, exogenas_futuras])
    else:
        exog_para_previsao = None

    # 3. Treina o modelo SARIMAX e gera as previsões.
    resultado_modelo, previsao_df = treinar_modelo_e_prever(treino_y, treino_X, dias_para_prever, exog_para_previsao)

    # 4. Simula o estoque e gera o dashboard com todas as informações compiladas.
    nome_dashboard = simular_estoque_e_gerar_dashboard(treino_y, teste_y, serie_completa, previsao_df, ID_PRODUTO, resultado_modelo)

    # --- Saída Final ---
    print(f"\nDashboard gerado com sucesso: {nome_dashboard}")