# Desafio de dados - Aignosi

## Passo-a-passo seguido

### Leitura do kaggle e entendimento do desafio
Após a leitura do tema do desafio, uma análise exploratória de séries temporais multivariadas, busquei entender mais sobre o assunto, pois este era um tema absolutamente novo. As principais dificuldades foram descobrir quais as técnicas mais adequadas para demonstrar o bom potencial preditivo dos dados. O primeiro passo foi entender o funcionamento dos sensores, a taxa de aquisição e o objeto de medição de cada um deles. Após isto, surgiu a ideia de um template para facilitar a análise: visãoo geral seguida de uma análise baseada em cada grupo de variáveis, da forma em que foram apresentadas no kaggle:
1.  Qualidade pré-processamento
2.  Variáveis importantes para o processo minerador
3.  Outras variáveis ambientais de interesse
4.  % de Ferro e target: % Silica pós-processamento

### Processamento dos dados
Apesar da alta usabilidade do dataset, foram necessários alguns procedimentos para possibilitar a análise. O primeiro foi trocar a separação de decimais de vírgula por pontos e converter os números para float (estavam em formato pandas object). As datas, além disso, apresentavam valores repetidos: só marcavam as horas. Incluí nas horas a maior frequência de aquisição de acordo com o kaggle, de 20s. Outra questão foi o período de anormalidade no mês de março, em que os sensores registraram um valor médio e não apresentavam variações. Optei por retirar o período anterior a abril por ser pequeno o suficiente com relação ao dataset e grande demais para imputar valores, considerando 1 mês de valores imputados de sensores que operam em período de segundos e horas.

### Análise exploratória
Primeiramente, foram levantados histogramas para analisar como os dados estão distribuídos. Uma distribuição normal pode implicar em modelos mais robustos como, por exemplo, redes neurais que utilizam gradiente descendente, onde valores muito altos ou muito baixos podem enviesar o desempenho. Além disso, distribuições bem definidas podem ser úteis para a detecção de ruídos e anormalidades.
De forma a aprofundar a análise de outliers, foram realizados boxplots para representar melhor a presença de outliers. Esta detecção é importante, pois valores anormais e ruídos podem danificar o desempenho do modelo.
Após esta visão geral, me aprofundei em cada grupo de dados. As análises de cada grupo seguiram um padrão:
1. Identificação de tendências e sazonalidades: a presença ou não de tendências e sazonalidades influencia diretamente na escolha de qual modelo escolher. Modelos estatísticos mais clássicos, como o ARIMA, exigem estacionariedade dos dados, enquanto modelos baseados em aprendizado de máquina são mais robustos e podem utilizar sazonalidades e tendências para detectar padrões que aumentarm a previsibilidade da série temporal.
2. Teste de estacionariedade: Escolhi realizar um teste estatístico, ADF, para não depender apenas da análise visual das séries, mas sim ter uma estatística mais confiável. O resultado confirmou as análises visuais para a maioria das séries, mas para a Ore Pulp pH, que imaginei ter uma tendência ascendente, o teste indicou estacionariedade.
3. Correlação entre variáveis: a análise de correlação entre variáveis foi realizada para evitar redundâncias e multicolinearidades nos modelos, o que poderia causar possível overfitting. Para além disto, a redundância detectada em algumas das features me deu um insight sobre a otimização do processo minerador, pois o sinal de um sensor pode ser usado para validar o funcionamento de outro/os por meio de análises que envolvem ciência de dados. Tal correlação, entretanto, se confirmou em poucas features: % Iron Feed e % Silica Feed, Column 01 e 03 Air Flow e Column 02 e 03 Air Flow.
4. Correlação com o target: Realizei o plot de correlação cruzada entre as features e o % Silica Concentrate em busca de possíveis correlações que indicassem um maior potencial preditivo nos dados. Os resultados, entretanto, foram desanimadores. A única variável com correlação considerável foi o % Iron Concentrate, porém um dos objetivos do desafio no kaggle seria prever o target sem o uso desta variável em específico.
5. Autocorrelação do target: realizei um plot de autocorrelação do target com a expectativa de visualizar padrões sazonais que viabilizassem um melhor potencial preditivo nos dados, porém o resultado não foi animador e não ultrapassou 0.2.

### Construção de modelo
Apesar de ser um etapa opcional, optei por construir um modelo XGBoost com o objetivo de visualizar as variáveis mais importantes para a previsão do modelo, e não focando unicamente no desempenho. Este foi razoável, apresentando uma taxa de erro de aproximadamente 29%, comparando-se o RMSE com o valor médio do target.

### Apresentação
Na apresentação, procurei ressaltar apenas os pontos positivos encontrados nos dados, que foram muitos, deixando de lado características desanimadoras, como baixa correlação entre features e targets. Tais características não representam empecilho para a construção de um modelo robusto e eficiente, mas julguei não ser interessante apresentá-las ao cliente hipotético, pelo menos em um primeira reunião.
