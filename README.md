<div align="center">
  <img src="https://github.com/user-attachments/assets/ccb6f5f1-0e07-4eb2-aa7c-5f681c57a59c" alt="Descrição da imagem" width="1000"/>
</div>

<h1 align="center">Tarrasque 🧌</h1>

<h2 align="center">Previsão de severidade de enfisema</h2>

<h4 align="center"><strong>Autores:</strong> Júlia Guedes, Lorena Ribeiro e Emily Gomes</h4>

<h4 align="center"><strong>Professor:</strong> Daniel R. Cassar</h4>


## 📝 Descrição
<p align="justify"> 
De forma geral, é possível definir otimização como uma abordagem matemática voltada para a busca da melhor solução possível dentro de um conjunto de alternativas. Sendo que, para isso, é preciso respeitar os objetivos e restrições impostos pelo problema. Atualmente, diversas estratégias de otimização estão disponíveis, sendo sua aplicação dependente da natureza e complexidade do cenário analisado [1]. No contexto do aprendizado de máquina, emergem 3 principais formas de otimização de hiperparâmetros: Busca Aleatória, busca em grade e otimização bayesiana (Optuna). Para esse trabalho, serão considerados como hiperparâmetros, serão considerados a função de ativação, número de camadas, quantidade de neurônios por camada e taxa de aprendizado.
</p>

<p align="justify">
Esse repositório busca explorar essas três formas de otimização para a construção de uma rede neural <em> Multilayer Perceptron</em> (MLP) classificadora multiclasse. Para a construção do problema, o dataset escolhido foi o "CD4/CD8 Ratio: T helper cells", do módulo Kaggle, o qual trata sobre severidade de enfisemas pulmonares em pessoas infectadas com HIV. Como algumas <em>features</em>, temos características do pacientes analisados, tais como idade, gênero, indíce de massa corporal, uso de drogas e marcadores relacionados a essas doenças presentes no sangue e no plasma, como CD14 solúvel, volume expiratório forçado (FEV1) e capacidade de difusão (DLCO). 
</p>

## 📔 Notebooks e arquivos do projeto
* `Tarrasque.zip`: Notebook principal que implementa os códigos de otimizações com as diferentes técnicas, bem como aprofunda os conceitos teóricos envolvidos e analisa os resultados obtidos por cada tipo de otimização.
* `imagens`: Contém a logo utilizada no Notebook principal e no READ ME do Github ("logo_ilum-CNPEM.png")

## 🪼 Métodos para a busca de hiperparâmetros
**Busca aleatória**
<p align="justify">
A <strong>busca aleatória</strong>, por sua vez, utiliza distribuições estatísticas em vez de valores discretos para os parâmetros. Ao longo das iterações, os modelos são comparados com o objetivo de encontrar a melhor configuração. Nessa metodologia, não há garantia de que o melhor modelo será encontrado, mas, geralmente, ela retorna resultados comparáveis aos da busca em grade, com menor tempo de execução. [2]</p>
  
<p align="justify">
Para a realização desse tipo de busca, o módulo ``Optuna``, definido no modo aleatório ``(sampler=optuna.samplers.RandomSampler(seed=51012))``, foi implementado. A partir do resultado da melhor arquitetura, essa foi testada para a rede MLP, em que dentre as análises foi apresentada a seguinte matriz de confusão:</p>

<p align="justify">
  <img src="https://github.com/user-attachments/assets/f100eb94-d1b2-48d3-95d8-cb9172f67e23" alt="image" style="display: block; margin: auto; text-align: justify;">
</p>


**Busca em grade**
<p align="justify">
A <strong>busca em grade</strong> é uma metodologia para o ajuste de hiperparâmetros que consiste em explorar exaustivamente o espaço com todos os conjuntos possíveis de hiperparâmetros. Com isso, o objetivo é encontrar o melhor conjunto dessas variáveis para o modelo. Contudo, ao gerar todas as configurações possíveis de valores discretos, há um alto consumo de recursos computacionais e, desse modo, essa abordagem mostra-se ineficiente para lidar com a maioria dos problemas. [2]

<p align="justify">
Para a definição desse tipo de busca, o módulo `GridSearchCV` da biblioteca ``Scikit-Learn`` foi utilizado. A partir da arquitetura com melhor desempenho encontrado, essa foi aplicada posteriormente na rede, avaliando sua acurácia, e gerando a matriz de confusão desse modelo:

![image](https://github.com/user-attachments/assets/e857ecdc-6cbd-4ea2-b630-81491dcb99bd)

</p>

**Otimização bayesiana**
<p style="text-align:justify;">
Em relação ao <strong>Optuna</strong>, esse algoritmo utiliza o princípio do Teorema de Bayes para encontrar os hiperparâmetros. Ou seja, o processo é iterativo, sendo que o próximo palpite para a combinação de variáveis depende da anterior. A partir disso, são selecionados de forma probabilística um novo conjunto de valores de hiperparâmetros com maior probabilidade de gerar melhores resultados. 
</p>

<p align="justify">
Finalmente, em relação a essa forma de otimização, o modo clássico do módulo ``Optuna`` foi utilizado. Gerando a melhor arquitetura de rede encontrada, que foi testada na MLP classificadora, resultando na matriz de confusão apresentada a seguir:
</p>

![image](https://github.com/user-attachments/assets/ba9da3be-ec97-4f31-a251-2703c011fb9d)


## 😁 Conclusão

<p align="justify">
Ao final do projeto, foi possível otimizar os parâmetros de uma rede MLP classificadora multiclasse, a qual visava a previsão da severidade de enfisemas pulmonares em pacientes com HIV. A partir dos três métodos de otimização apicados, foi possível obter três conjuntos de hiperparâmetros. Vale ressaltar que os hiperparâmetros dos métodos de busca em grade e busca bayesiana foram definidos dentro de um intervalo possível, enquanto a busca em grade foi feita com valores de hiperparâmetros definidos de forma discreta e sem dropout. A partir do treinamento e teste das redes MLP com cada uma das combinações de parâmetros, obteve-se diferentes valores de acurácia, sendo os dois maiores valores de 41.18% com a busca aleatória e bayesiana. Quanto ao menor resultado, foi a acurácia de 35.29%, com a busca em grade. Quanto a matriz de confusão tanto a busca em grade, quanto busca aleatória apresentam apenas dois valores na diagonal principal, enquanto a busca bayesiana apresenta três. Vale ressaltar também que em todas, as predições mais fidedignas foram em relação a classe 0, que apresenta maior frequência de dados. Apesar dos valores iguais entre a acurácia do modelo Bayesiano e de Busca aleatória, as matrizes de confusão foram diferentes o que demonstra modelos diferentes em suas predições.  Portanto, a acurácia próxima ao baseline, pode ter ocorrido devido ao desbalanceamento das classes. Além disso, foi possível observar que a aplicação da ferramenta de parada antecipada foi uma ferramenta que auxilixou na redução do custo computacional .Desse modo, foi possível testar métodos diferentes de otimizadores e avaliar o desempenho de cada um deles para a predição desse modelo. 
</p>

## 🖇️ Informações técnicas
* Linguagem de programação: `Python 3.9`
* Software:  `Jupyter Notebook`

* Bibliotecas e Módulos: `random` `zipfile` `os` `pandas` `scikit-learn` `skorch` `pandas` `numpy` `itertools` `torch` `optuna` `copy`
<br>


## 👩‍🦳 Referências
[1] [What Is Optimization Modeling? | IBM](https://www.ibm.com/think/topics/optimization-model). Acesso em: 4 maio 2025.

[2] [What Is Hyperparameter Tuning? | IBM](https://www.ibm.com/think/topics/hyperparameter-tuning). Acesso em: 7 maio 2025.

[3] [SKORCH – Quickstart: Grid Search](https://skorch.readthedocs.io/en/stable/user/quickstart.html#grid-search). Acesso em: 3 jun. 2025.

[4] [Quickstart — skorch 1.1.0 documentation](https://skorch.readthedocs.io/en/stable/user/quickstart.html#grid-search). Acesso em: 3 jun. 2025.

[5] [Tuning MLP by using Optuna – Gist by toshihikoyanase](https://gist.github.com/toshihikoyanase/7e75b054395ec0a6dbb144a300862f60). Acesso em: 28 mai. 2025.

[6] [scikit-learn hyperparameter optimization for MLPClassifier – PANJEH (Medium)](https://panjeh.medium.com/scikit-learn-hyperparameter-optimization-for-mlpclassifier-4d670413042b). Acesso em: 21 mai. 2025.

 ## 🧠 Contribuições dos Colaboradores
| GitHub | Contribuições |
|:-----|:--------------|
| [Júlia Guedes A. dos Santos](https://github.com/JuliaGuedesASantos) | Introdução, otimização de modelos (bayesiano e busca aleatória), treinamento final dos modelos, comentários no código e READ ME |
| [Lorena Ribeiro Nascimento](https://github.com/lorena881) | Introdução, otimização de modelos (bayesiano e busca aleatória), comentários no código, READ ME | (https://github.com/MEmilyGomes)
| [Maria Emily Nayla Gomes da Silva](https://github.com/lorena881) | Introdução, otimização de modelos (busca em grade), READ ME e comentários no código |
| [Daniel Roberto Cassar](https://github.com/drcassar) | Orientador |
 
