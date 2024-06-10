# Exemplo de CLASSIFICAÇÃO

Este repositório contém exemplos de implementações de algoritmos de classificação de e-mails em Python utilizando a biblioteca `scikit-learn`. Os algoritmos implementados são: Naive Bayes Gaussiano, Naive Bayes Multinomial, K-Nearest Neighbors (KNN), Support Vector Machine (SVM) e Árvore de Decisão.

### Tipos de Classificação:

1. **Naive Bayes**:
   - **Gaussiano**: Usa o algoritmo Naive Bayes Gaussiano para classificar e-mails.
   - **Multinomial**: Utiliza o algoritmo Naive Bayes Multinomial para classificação.

2. **K-Nearest Neighbors (KNN)**:
   - Usa o algoritmo KNN para classificar e-mails com base na proximidade com os vizinhos mais próximos.

3. **Support Vector Machine (SVM)**:
   - Utiliza o algoritmo SVM para classificar e-mails.

4. **Árvore de Decisão**:
   - Usa o algoritmo de Árvore de Decisão para classificar e-mails.

# Resultados

### Naives Bayes

**Gaussiano**

~~~
# Técnica Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Dados de exemplo
x = [[100, 20], [150, 30], [120, 25], [140, 28]]
y = ['Não Spam', 'Spam', 'Não Spam', 'Spam']

# Treinando o modelo
model = GaussianNB()
model.fit(x, y)

# Previsão para um novo e-mail
novo_email = [[130, 22]]
resultado = model.predict(novo_email)
print(f"Previsão para o novo e-mail: {resultado[0]}")
~~~
![bayes gaussiano](https://github.com/renebttg/Exemplo_Classificacao_Regressao/assets/114888521/885a4757-3987-47f1-b9d2-ee1ff67fd832)

**Multinominal**

~~~

# Técnica Naive Bayes
# Passo 1: Importar as bibliotecas necessárias
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Passo 2: Preparar os dados
emails = [ "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
           "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
           "Confira as novas ofertas da loja. Não perca!",
           "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
           "Lembrete: pagamento da fatura do cartão de crédito vence amanhã."
]

labels = [1, 1, 1, 0, 0] # 1 para spam, 0 para não spam

# Passo 3: Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# Passo 4: Dividir os dads em conjunto de treinamento e conjunto de teste
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Passo 5: Criar e treinar o modelo
model = MultinomialNB() # Criar o modelo Naive Bayes multinomial
model.fit(x_train, y_train) # Treinar o modelo com os dados de treinamento

# Passo 6: Fazer previsões
predictions = model.predict(x_test) # Fazer previsões usando o conjunto de teste

# Passo 7: Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions) # Caleular a precisão do modelo
print("Accuracy: ", accuracy)
~~~
![Multinominal](https://github.com/renebttg/Exemplo_Classificacao_Regressao/assets/114888521/c5f2610f-3837-4556-8a81-08707fc715fc)

### K-Nearest Neighbors (KNN)

~~~
# Técnica K-Nearest Neighbors (KNN)
# Passo 1: Importar as bibliotecas necessárias
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Passo 2: Preparar os dados
emails = [ "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
           "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
           "Confira as novas ofertas da loja. Não perca!",
           "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
           "Lembrete: pagamento da fatura do cartão de crédito vence amanhã."
]

labels = [1, 1, 1, 0, 0] # 1 para spam, 0 para não spam

# Passo 3: Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# Passo 4: Dividir os dads em conjunto de treinamento e conjunto de teste
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Passo 5: Criar e treinar o modelo
model = KNeighborsClassifier(n_neighbors=3) # Criar o modelo KNN com 3 vizinhos
model.fit(x_train, y_train) # Treinar o modelo com os dados de treinamento

# Passo 6: Fazer previsões
predictions = model.predict(x_test) # Fazer previsões usando o conjunto de teste

# Passo 7: Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions) # Calcular a precisão do modelo
print("Accuracy: ", accuracy)
~~~
![KNN](https://github.com/renebttg/Exemplo_Classificacao_Regressao/assets/114888521/da46a701-9e60-494a-9f6a-c5c74faad9e2)

### Support Vector Machine (SVM)

~~~
# Técnica Support Vector Machine (SVM)
# Passo 1: Importar as bibliotecas necessárias
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Passo 2: Preparar os dados
emails = [ "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
           "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
           "Confira as novas ofertas da loja. Não perca!",
           "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
           "Lembrete: pagamento da fatura do cartão de crédito vence amanhã."
]

labels = [1, 1, 1, 0, 0] # 1 para spam, 0 para não spam

# Passo 3: Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# Passo 4: Dividir os dads em conjunto de treinamento e conjunto de teste
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Passo 5: Criar e treinar o modelo
model = SVC(kernel='linear') # Criar o modelo SVM com kernel linear
model.fit(x_train, y_train) # Treinar o modelo com os dados de treinamento

# Passo 6: Fazer previsões
predictions = model.predict(x_test) # Fazer previsões usando o conjunto de teste

# Passo 7: Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions) # Caleular a precisão do modelo
print("Accuracy: ", accuracy)
~~~
![SVM](https://github.com/renebttg/Exemplo_Classificacao_Regressao/assets/114888521/db4450fa-46c9-4924-a405-e8f6de125e19)

### Árvore de Decisão

~~~
# Técnica Árvore de Decisão
# Passo 1: Importar as bibliotecas necessárias
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Passo 2: Preparar os dados
emails = [ "Oferta imperdível! Ganhe 50% de desconto em todos os produtos!",
           "Você ganhou um prêmio de R$ 10.000! Clique aqui para resgatar.",
           "Confira as novas ofertas da loja. Não perca!",
           "Reunião de equipe amanhã às 10h. Por favor, confirme sua presença.",
           "Lembrete: pagamento da fatura do cartão de crédito vence amanhã."
]

labels = [1, 1, 1, 0, 0] # 1 para spam, 0 para não spam

# Passo 3: Transformar os dados em uma matriz de contagem de palavras (bag of words)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# Passo 4: Dividir os dads em conjunto de treinamento e conjunto de teste
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Passo 5: Criar e treinar o modelo
model = DecisionTreeClassifier() # Criar o modelo de Àrvore de Decisão
model.fit(x_train, y_train) # Treinar o modelo com os dados de treinamento

# Passo 6: Fazer previsões
predictions = model.predict(x_test) # Fazer previsões usando o conjunto de teste

# Passo 7: Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions) # Caleular a precisão do modelo
print("Accuracy: ", accuracy)
~~~
![TREE](https://github.com/renebttg/Exemplo_Classificacao_Regressao/assets/114888521/039dd6f1-2d42-4f01-ade4-d7325f2991d3)

## Exemplo Regressão

Bem-vindo ao repositório "Exemplo Regressão"! Aqui você encontrará implementações de diversos algoritmos de regressão em Python, utilizando a biblioteca `scikit-learn`. A regressão é uma técnica estatística poderosa usada para modelar e analisar relacionamentos entre variáveis. Este repositório oferece exemplos práticos de diferentes tipos de regressão, desde a simples regressão linear até métodos mais avançados de regressão não linear.

### Tipos de Regressão:

- **Regressão Linear Simples**: Este exemplo demonstra como realizar uma regressão linear simples para prever notas de exames com base no número de horas de estudo.
- **Regressão Linear Múltipla**: Implementa a regressão linear múltipla para prever notas de exames com base no número de horas de estudo e no tempo de sono.
- **Regressão Linear Logística**: Apresenta a aplicação da regressão linear logística para classificar espécies de íris com base em características da flor.
- **Regressão Polinomial**: Demonstração de como ajustar uma curva polinomial aos dados para modelar relacionamentos não lineares.
- **Métodos de Regressão Não Linear**: Implementa métodos avançados de regressão não linear, como ajuste de curvas exponenciais.

# Resultados:

A seguir estão os códigos de implementação para cada tipo de regressão:

### Regressão Linear

~~~
# Regressão Linear
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## Dados de horas de estudo e notas do exame
horas_estudo = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).reshape(-1,1)
notas_exames = np.array([65, 70, 75, 80, 85, 90, 95, 100, 105, 110])

# Criar um modelo de regressão linear
modelo = LinearRegression()

# Treinar o modelo
modelo.fit(horas_estudo, notas_exames)

# Obter os coeficientes do modelo
coef_angular = modelo.coef_[0]
coef_linear = modelo.intercept_
# Plotar os dados e a reta de regressão
plt.scatter(horas_estudo, notas_exames, color='blue')
plt.plot(horas_estudo, modelo.predict(horas_estudo), color='red')
plt.title('Regressão Linear Simples')
plt.xlabel('Horas de estudo')
plt.ylabel('Notas no Exame')
plt.show()

# Fazer previsões com o modelo
horas_estudo_novo = np.array([[8]]) # Horas do estudo do novo aluno
nota_prevista = modelo.predict(horas_estudo_novo)
print("Nota prevista para {} horas de estudo: {:.2f}".format(horas_estudo_novo[0][0], nota_prevista[0]))
~~~
![Linear](https://github.com/renebttg/Exemplo_Classificacao_Regressao/assets/114888521/e1c44b4c-427d-4c7a-8478-4c6b9e721c5d)


### Regressão Linear Múltipla

~~~
# Regressão Linear Múltipla
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Dados de horas de estudo, tempo de sono e notas do exame
horas_estudo = np.array([2,3,4,5,6,7,8,9,10,11]).reshape(-1,1)
tempo_sono = np.array([7,6,5,6,7,8,9,8,7,6]).reshape(-1,1)
notas_exames = np.array([65,70,75,80,85,90,95,100,105,110])

# Criar um modelo de regressão linear
modelo = LinearRegression()

# Combinação de horas de estudos e tempo de sono com variáveis independentes
x = np.concatenate((horas_estudo,tempo_sono), axis=1)

# Treinar o modelo
modelo.fit(x, notas_exames)

# Coeficientes do modelo
coef_angular = modelo.coef_
coef_linear = modelo.intercept_

# Plotar os dados em 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(horas_estudo, tempo_sono, notas_exames, color='blue')

# Prever notas para o intervalo de horas de estudos e tempo de sono
x_test = np.array([[x,y] for x in range(2, 12) for y in range(5,10)])
nota_previstas = modelo.predict(x_test)

# Plotar o plano de regressão
x_surf, y_surf = np.meshgrid(range(2,12), range(5,10))
exog = np.column_stack((x_surf.flatten(), y_surf.flatten()))
nota_previstas = modelo.predict(exog)
ax.plot_surface(x_surf, y_surf, nota_previstas.reshape(x_surf.shape), color='red', alpha=0.5)

ax.set_xlabel('Horas de Estudo')
ax.set_ylabel('Tempo de Sono')
ax.set_zlabel('Notas do Exame')

plt.show()
~~~
![Linear multipla](https://github.com/renebttg/Exemplo_Classificacao_Regressao/assets/114888521/c9ea752e-1e56-451f-8105-d7a89426467f)


### Regressão Linear Logística

~~~
# Regressão linear logística
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Passo 1: Carregar o conjunto de dados iris
iris = load_iris()
x = iris.data[:, :2] # Apenas as duas primeiras características para visualização
y = iris.target

# Passo 2: Dividir o conjunto de dados em conjunto de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=42)

# Passo 3: Pré-processamento (padronização)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Passo 4: Criar e treinar modelo de regressãologística
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# Passo 5: Fazer previsões no conjunto de teste
y_pred = model.predict(x_test_scaled)

# Passo 6: Avaliar o desempenho do modelo
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test,y_pred))

# Passo 7: Visiuazlização dos resultados
plt.figure(figsize=(10,6))

# Plotar os pontos de dados de treinamento
plt.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], c=y_train, cmap='viridis', edgecolors='k', label='Treinamento')

# Plotar os pontos de dados de teste
plt.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], c=y_test, cmap='viridis', marker='x', s=100, label='Teste')

# Plotar as regiões de decisão
x_min, x_max = x_train_scaled[:, 0].min() -1, x_train_scaled[:,0].max() + 1
y_min, y_max = x_train_scaled[:, 0].min() -1, x_train_scaled[:,0].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
plt.contourf(xx, yy, z, alpha=0.3, cmap='viridis')

plt.xlabel('Comprimento da Sépala Padronizado')
plt.ylabel('Largura da Sépala Padronizado')
plt.title('Regressão Logística para Classificaçãode Espécies Iris')
plt.legend()
plt.show()
~~~
![Logistica](https://github.com/renebttg/Exemplo_Classificacao_Regressao/assets/114888521/8ee5cc8d-9727-4106-ad19-7c199525c2dd)


### Regressão Polinomial

~~~
# Regressão Polinomial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Gerar dados sintéticos
np.random.seed(0)
x = 2 * np.random.rand(100, 1) -1 # Variáveis independentes entre -1 e 1
y = 3 * x**2 + 0.5 * x + 2 + np.random.randn(100, 1) # Relação quadrática com ruído

# Plotar os dados
plt.scatter(x,y, color='blue', label='Dados')

# Ajustar uma regressão polimonial de grau 2 aos dados
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

# Plotar a curva ajustada
x_plot = np.linspace(-1,1,100).reshape(-1,1)
x_plot_poly = poly_features.transform(x_plot)
y_plot = lin_reg.predict(x_plot_poly)
plt.plot(x_plot, y_plot, color='red', label='Regressão Polimonial (grau 2) ')

# Avaliar Modelo
y_pred = lin_reg.predict(x_poly)
mse = mean_squared_error(y,y_pred)
print("Erro médio quadrático:", mse)

plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')
plt.title('Regressão Polimonial de Grau 2')
plt.legend()
plt.show()
~~~
![Polimonial](https://github.com/renebttg/Exemplo_Classificacao_Regressao/assets/114888521/1fd2ed0f-1d1e-43ca-9f31-81eb6365ab1c)


### Métodos de Regressão Não Linear

~~~
# Métodos de Regressão Não Linear
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# Função exponencial para ajustar aos dados
def modelo_exponencial(x, a, b):
    return a * np.exp(b * x)

# Gerar dados sintéticos
np.random.seed(0)
x = np.linspace(0, 5, 100) # Variável independente
y = 2.5 * np.exp(0.5 * x) + np.random.normal(0, 0.5,100) # Relação exponencial com ruído

# Ajustar o modelo aos dados usando o curve_fit
params, _ = curve_fit(modelo_exponencial,x,y)

# Plotar os dados
plt.scatter(x,y, color='blue', label='Dados')

# Plotar a curva ajustada
plt.plot(x, modelo_exponencial(x, *params), color='red', label='Regressão Exponencial')

plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')
plt.title('Regressão Não Linear Exponencial')
plt.legend()
plt.show()
~~~
![Não Linear](https://github.com/renebttg/Exemplo_Classificacao_Regressao/assets/114888521/f208d69f-1e3a-4409-a9cd-ff258fc5d3a4)


