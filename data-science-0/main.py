#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[38]:


import pandas as pd
import numpy as np
get_ipython().system('pip install sklearn')


# In[39]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# In[4]:


black_friday.shape


# In[5]:


#conferindo as colunas
black_friday.columns


# In[6]:


#conferindo os tipos de variável nas colunas
black_friday.dtypes


# In[7]:


#conferindo os dados gerais do DF
black_friday.info()


# In[8]:


#mudando o nome do DF original para mexer sem estragar
dados = black_friday
dados.head()


# ### Questão 1: Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple (n_observacoes, n_colunas).

# In[9]:


#usando o shape do pandas
dados.shape


# ### Questão 2: Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[10]:


#verificando qual a porcentagem de mulheres e homens, just because
#o dataset tem aproximadamente 75% de homens e 25% de mulheres
dados['Gender'].value_counts(normalize=True)


# In[11]:


#conferindo os tipos de Age
dados['Age'].unique()


# In[12]:


#fazendo uma query do Pandas
mulheres_jovens = dados.query("Gender == 'F' and Age == '26-35'")
mulheres_jovens.shape[0]


# In[13]:


#incluindo a resposta numa variável
q2 = mulheres_jovens.shape[0]
q2


# ### Questão 3: Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[14]:


#somando os usuários únicos usando a função nunique do pandas na coluna User_ID
#dados['User_ID'].nunique()
#colocando a resposta numa variável
q3 = dados['User_ID'].nunique()
q3


# ### Questão 4: Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[15]:


#tentando usar o nunique com o dtypes
dados.dtypes.nunique()


# In[16]:


#incluindo numa variável
q4 = dados.dtypes.nunique()
q4


# ### Questão 5: Qual porcentagem dos registros possui ao menos um valor null (None, ǸaN etc)? Responda como um único escalar entre 0 e 1.

# In[17]:


#vamos ver um trem aqui
dados.isna()


# In[18]:


#contando os dados não nulos usando o count
nao_nulos = dados.count().sum()
nao_nulos


# In[19]:


#somando os dados nulos - o primeiro sum soma por coluna, o segundo soma tudo
nulos = dados.isna().sum().sum()
nulos


# In[20]:


#compilando
total_de_dados = nulos + nao_nulos 
total_de_dados


# In[21]:


#dividindo os nulos pelo total devo ter uma resposta em float
q5 = nulos/total_de_dados
q5


# ### Questão 6: Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[22]:


#contando dados não nulos novamente
dados.count()


# In[23]:


#conferindo o valor mínimo de dados não nulos
dados.count().min()


# In[24]:


#contando nulos na coluna "Product_Category_3"
dados['Product_Category_3'].isna().sum()


# In[25]:


#colocando na variável
q6 = dados['Product_Category_3'].isna().sum()
q6


# ### Questão 7: Qual o valor mais frequente (sem contar nulls) em Product_Category_3? Responda como um único escalar.

# In[26]:


#usando a moda
dados['Product_Category_3'].mode()[0]


# In[27]:


q7 = dados['Product_Category_3'].mode()[0]
q7


# ### Questão 8: Qual a nova média da variável (coluna) Purchase após sua normalização? Responda como um único escalar.

# In[28]:


#visualizando a coluna
dados['Purchase']


# In[29]:


#contando os nulos
dados['Purchase'].isna().sum()


# In[30]:


#plotar a distribuição da variável
dados['Purchase'].hist()


# In[31]:


#conferindo a média antes
media = dados['Purchase'].mean()
media


# In[32]:


desvio_padrao = dados['Purchase'].std()


# In[33]:


#normalizando pela média (roubartilhado diretamente do StackOverflow)
novos_dados = dados['Purchase']-media/desvio_padrao
novos_dados


# In[34]:


#nova média - normalizando pela média
novos_dados.mean()


# In[35]:


#incluindo na variável
q8 = novos_dados.mean()
q8


# In[50]:


#normalizando com min max direto com o Pandas = normalized_df=(df-df.min())/(df.max()-df.min())
#criando variável para ficar mais legível
compras = dados['Purchase']
minimo = compras.min()
maximo = compras.max()

compras_normalizado = (compras - minimo)/(maximo - minimo)
compras_normalizado


# In[51]:


#média normalizada por mínimo e máximo
compras_normalizado.mean()


# In[52]:


#considerando a questão 9, vamos colocar esta resposta como a oficial
q8 = compras_normalizado.mean()
q8


# ### Questão 9: Quantas ocorrências entre -1 e 1 inclusive existem da variáel Purchase após sua padronização? Responda como um único escalar.

# In[55]:


compras = pd.DataFrame(compras_normalizado)
compras


# In[59]:


compras.describe()


# In[61]:


compras.isna().sum()


# In[63]:


#se o valor mínimo é 0 e o valor máximo é 1, então todos os valores estão entre -1 e 1
q9 = compras.count()[0]
q9


# ### Questão 10: Podemos afirmar que se uma observação é null em Product_Category_2 ela também o é em Product_Category_3? Responda com um bool (True, False).

# In[68]:


#solução roubartilhada
dados_comparacao = dados[dados['Product_Category_2'].isna()]
dados_comparacao
   


# In[70]:


dados_comparacao['Product_Category_2'].equals(dados_comparacao['Product_Category_3'])


# In[72]:


q10 = dados_comparacao['Product_Category_2'].equals(dados_comparacao['Product_Category_3'])
q10


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[ ]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[ ]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return q2


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[ ]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return q3


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[ ]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return q4


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[ ]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return q5


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[ ]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return q6


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[ ]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return q7


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[ ]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return q8


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[ ]:


def q9():
    # Retorne aqui o resultado da questão 9.
    return q9


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[ ]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return q10

