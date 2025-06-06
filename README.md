<div align="center">
  <img src="https://github.com/user-attachments/assets/ccb6f5f1-0e07-4eb2-aa7c-5f681c57a59c" alt="Descrição da imagem" width="1000"/>
</div>

<h1 align="center">Tarrasque 🧌</h1>

<h2 align="center">Previsão de severidade de enfisema</h2>

<h4 align="center"><strong>Autores:</strong> Júlia Guedes, Lorena Ribeiro e Emily Gomes</h4>

<h4 align="center"><strong>Professor:</strong> Daniel R. Cassar</h4>

<p align="center">
<img loading="lazy" src="http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge"/>
</p>

## 📝 Descrição
<p align="justify"> 
De forma geral, é possível definir otimização como uma abordagem matemática voltada para a busca da melhor solução possível dentro de um conjunto de alternativas. Sendo que, para isso, é preciso respeitar os objetivos e restrições impostos pelo problema. Atualmente, diversas estratégias de otimização estão disponíveis, sendo sua aplicação dependente da natureza e complexidade do cenário analisado [1]. No contexto do aprendizado de máquina, emergem 3 principais formas de otimização de hiperparâmetros: Busca Aleatória, busca em grade e otimização bayesiana (Optuna). Para esse trabalho, serão considerados como hiperparâmetros, serão considerados a função de ativação, número de camadas, quantidade de neurônios por camada e taxa de aprendizado.
</p>

<p align="justify">
Esse repositório busca explorar essas três formas de otimização para a construção de uma rede neural <em> Multilayer Perceptron</em> (MLP) classificadora multiclasse. Para a construção do problema, o dataset escolhido foi o "CD4/CD8 Ratio: T helper cells", do módulo Kaggle, o qual trata sobre severidade de enfisemas pulmonares em pessoas infectadas com HIV. Como algumas <em>features</em>, temos características do pacientes analisados, tais como idade, gênero, indíce de massa corporal, uso de drogas e marcadores relacionados a essas doenças presentes no sangue e no plasma, como CD14 solúvel, volume expiratório forçado (FEV1) e capacidade de difusão (DLCO). 
</p>

## 📔 Notebooks e arquivos do projeto
* `ATP -5.1 Tiamat.ipynb`: Notebook principal que implementa os códigos de otimizações com as diferentes técnicas, bem como aprofunda os conceitos teóricos envolvidos e analisa os resultados obtidos por cada tipo de otimização.
* `imagens`: Contém a logo utilizada no Notebook principal e no READ ME do Github ("logo_ilum-CNPEM.png")

## 🪼 Métodos para a busca de hiperparâmetros
**Busca aleatória**
<p align="justify">
A <strong>busca aleatória</strong>, por sua vez, utiliza distribuições estatísticas em vez de valores discretos para os parâmetros. Ao longo das iterações, os modelos são comparados com o objetivo de encontrar a melhor configuração. Nessa metodologia, não há garantia de que o melhor modelo será encontrado, mas, geralmente, ela retorna resultados comparáveis aos da busca em grade, com menor tempo de execução. [2]
</p>

Para a realização desse tipo de busca, o módulo ``Optuna``, definido no modo aleatório ``(sampler=optuna.samplers.RandomSampler(seed=51012))``, foi utilizado, a partir do seguinte código:

````python

from sklearn.model_selection import GridSearchCV


class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=nn.ReLU(), num_layers=2):
        super().__init__()
        self.nonlin = nonlin

        layers = []
        input_size = 30  # número de features do X
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, num_units))
            layers.append(nonlin)
            input_size = num_units  # as próximas camadas recebem `num_units`

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(num_units, QUANTIDADE_TARGET)

    def forward(self, X, **kwargs):
        X = self.hidden(X)
        X = self.output(X)
        return X
    
net = NeuralNetClassifier(
    MyModule,
    max_epochs=100,
    criterion=nn.CrossEntropyLoss(),
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)
    
params = {
    'lr': TAXA_APRENDIZADO,
    'module__num_units': NEURONIOS,         # por exemplo [10, 20, 50]
    'module__nonlin': [nn.ReLU(), nn.Sigmoid()],
    'module__num_layers': [1, 2, 3, 4]       # experimenta diferentes profundidades
}
gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

X_treino = X_treino.astype(np.float32) #?
y_treino = y_treino.astype(np.int64)
gs.fit(X_treino, y_treino)
print(gs.best_score_, gs.best_params_)
````


**Busca em grade**
<p align="justify">
A <strong>busca em grade</strong> é uma metodologia para o ajuste de hiperparâmetros que consiste em explorar exaustivamente o espaço com todos os conjuntos possíveis de hiperparâmetros. Com isso, o objetivo é encontrar o melhor conjunto dessas variáveis para o modelo. Contudo, ao gerar todas as configurações possíveis de valores discretos, há um alto consumo de recursos computacionais e, desse modo, essa abordagem mostra-se ineficiente para lidar com a maioria dos problemas. [2]

Para a definição desse tipo de busca, o módulo `GridSearchCV` da biblioteca ``Scikit-Learn`` foi utilizado, seguindo o pipeline abaixo:

````python

from sklearn.model_selection import GridSearchCV


class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=nn.ReLU(), num_layers=2):
        super().__init__()
        self.nonlin = nonlin

        layers = []
        input_size = 30  # número de features do X
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, num_units))
            layers.append(nonlin)
            input_size = num_units  # as próximas camadas recebem `num_units`

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(num_units, QUANTIDADE_TARGET)

    def forward(self, X, **kwargs):
        X = self.hidden(X)
        X = self.output(X)
        return X
    
net = NeuralNetClassifier(
    MyModule,
    max_epochs=100,
    criterion=nn.CrossEntropyLoss(),
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)
    
params = {
    'lr': TAXA_APRENDIZADO,
    'module__num_units': NEURONIOS,         # por exemplo [10, 20, 50]
    'module__nonlin': [nn.ReLU(), nn.Sigmoid()],
    'module__num_layers': [1, 2, 3, 4]       # experimenta diferentes profundidades
}
gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

X_treino = X_treino.astype(np.float32) #?
y_treino = y_treino.astype(np.int64)
gs.fit(X_treino, y_treino)
print(gs.best_score_, gs.best_params_)
````

</p>

**Otimização bayesiana**
<p style="text-align:justify;">
Em relação ao <strong>Optuna</strong>, esse algoritmo utiliza o princípio do Teorema de Bayes para encontrar os hiperparâmetros. Ou seja, o processo é iterativo, sendo que o próximo palpite para a combinação de variáveis depende da anterior. A partir disso, são selecionados de forma probabilística um novo conjunto de valores de hiperparâmetros com maior probabilidade de gerar melhores resultados. 
</p>

Finalmente, em relação a essa forma de otimização, o modo clássico do módulo ``Optuna`` foi utilizado:
</p>

````python

from sklearn.model_selection import GridSearchCV


class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=nn.ReLU(), num_layers=2):
        super().__init__()
        self.nonlin = nonlin

        layers = []
        input_size = 30  # número de features do X
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, num_units))
            layers.append(nonlin)
            input_size = num_units  # as próximas camadas recebem `num_units`

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(num_units, QUANTIDADE_TARGET)

    def forward(self, X, **kwargs):
        X = self.hidden(X)
        X = self.output(X)
        return X
    
net = NeuralNetClassifier(
    MyModule,
    max_epochs=100,
    criterion=nn.CrossEntropyLoss(),
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)
    
params = {
    'lr': TAXA_APRENDIZADO,
    'module__num_units': NEURONIOS,         # por exemplo [10, 20, 50]
    'module__nonlin': [nn.ReLU(), nn.Sigmoid()],
    'module__num_layers': [1, 2, 3, 4]       # experimenta diferentes profundidades
}
gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

X_treino = X_treino.astype(np.float32) #?
y_treino = y_treino.astype(np.int64)
gs.fit(X_treino, y_treino)
print(gs.best_score_, gs.best_params_)

````


## 😁 Conclusão

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
| [Júlia Guedes A. dos Santos](https://github.com/JuliaGuedesASantos) | Introdução, otimização de modelos (bayesiano e busca aleatória), treinamento final dos modelos e READ ME |
| [Lorena Ribeiro Nascimento](https://github.com/lorena881) | Introdução, otimização de modelos (bayesiano e busca aleatória), comentários no código | (https://github.com/MEmilyGomes)
| [Maria Emily Nayla Gomes da Silva](https://github.com/lorena881) | Introdução, otimização de modelos (busca em grade), READ ME e comentários no código |
| [Daniel Roberto Cassar](https://github.com/drcassar) | Orientador |
 
