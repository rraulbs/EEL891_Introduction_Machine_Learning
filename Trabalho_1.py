# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:07:10 2020

@author: Raul
"""
#===============================================================================
#
#   EEL891 (Introdução ao Aprendizado de Máquina)
#   TRABALHO DE AVALIAÇÃO 01
#   RAUL BAPTISTA DE SOUZA - DRE: 115110845
#
#   Modelo preditivo de para apoio à decisão de aprovação de crédito.
#
#   Identifica, dentre os clientes que solicitam um produto de crédito 
#   (como um cartão de crédito ou um empréstimo pessoal, por exemplo) e 
#   que cumprem os pré-requisitos essenciais para a aprovação do crédito,
#    aqueles que apresentem alto risco de não conseguirem honrar o pagamento,
#    tornando-se inadimplentes. 
#   
#   Para isso, é fornecido um arquivo com dados históricos de 20.000 
#   solicitações de produtos de créditos que foram aprovadas pela instituição, 
#   acompanhadas do respectivo desfecho, ou seja, acompanhadas da indicação
#   de quais desses solicitantes conseguiram honrar os pagamentos 
#   (50% dos casos) e quais ficaram inadimplentes (50% dos casos).
#
#   Com base nesses dados históricos, o classificador, a partir dos dados
#   de uma nova solicitação de crédito, prediz se este solicitante será um
#   bom ou mau pagador.
#
#===============================================================================

#-------------------------------------------------------------------------------
# Importar bibliotecas
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
tf.__version__
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
# MLP for with n-fold cross validation:
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
# DRE = seed
seed = 115110845
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(seed)
# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(seed)
#%%
#-------------------------------------------------------------------------------
# Ler o arquivo CSV com os dados do conjunto aprovação de crédito
#-------------------------------------------------------------------------------
df_1        = pd.read_csv('conjunto_de_treinamento.csv')  
df_2        = pd.read_csv('conjunto_de_teste.csv')
df_train    = pd.read_csv('conjunto_de_treinamento.csv')    # Para treinar o modelo
df_test     = pd.read_csv('conjunto_de_teste.csv')          # Para submeter resutado na competição
#%%
#-------------------------------------------------------------------------------
# Verificar se há espaços vazios
#-------------------------------------------------------------------------------
print("-------------------------------------------------")
print("---Verificando se há dados ausentes nas colunas--")
print(df_1.isna().sum())
print("-------------------------------------------------")
#%%
#-------------------------------------------------------------------------------
# Pré-Processamento
#-------------------------------------------------------------------------------
def pre_processing(dataset, to_drop, to_onehot, to_bin):
    # drop columns
    dataset = dataset.drop(to_drop, axis=1)
    # one-hot encoding
    dataset = pd.get_dummies(dataset, columns = to_onehot)
    # binarizar
    binarizador = LabelBinarizer()
    for v in to_bin:
        dataset[v] = binarizador.fit_transform(dataset[v])
    
    return dataset
#-------------------------------------------------------------------------------
df_train['meses_na_residencia'] = df_train['meses_na_residencia'].fillna(df_train['meses_na_residencia'].median())
df_train['tipo_residencia'] = df_train['tipo_residencia'].fillna(1)
df_train['ocupacao'] = df_train['ocupacao'].fillna(2)
df_train['profissao'] = df_train['profissao'].fillna(9)
df_train['sexo'] = df_train['sexo'].apply(lambda r: r.replace(' ','N'))
df_train = pre_processing(dataset = df_train,
               to_drop = ['grau_instrucao',
                    'estado_onde_nasceu', 'estado_onde_reside',
                    'codigo_area_telefone_residencial',
                    'possui_telefone_celular',
                    'qtde_contas_bancarias_especiais', 'estado_onde_trabalha',
                    'codigo_area_telefone_trabalho', 'meses_no_trabalho',
                    'profissao_companheiro',
                    'grau_instrucao_companheiro', 'local_onde_reside',
                    'local_onde_trabalha'
                    ], 
               to_onehot = ['estado_civil', 'produto_solicitado', 'dia_vencimento',
                                      'forma_envio_solicitacao',
                                      'nacionalidade', 'sexo', 'tipo_residencia',
                                      'profissao', 'ocupacao', 'qtde_contas_bancarias'], 
               to_bin = ['tipo_endereco','possui_telefone_residencial',
                         'vinculo_formal_com_empresa', 
                         'possui_telefone_trabalho',
                         'inadimplente'], 
               )
df_test['meses_na_residencia'] = df_test['meses_na_residencia'].fillna(df_test['meses_na_residencia'].median())
df_test['tipo_residencia'] = df_test['tipo_residencia'].fillna(1)
df_test['ocupacao'] = df_test['ocupacao'].fillna(2)
df_test['profissao'] = df_test['profissao'].fillna(9)
df_test['sexo'] = df_test['sexo'].apply(lambda r: r.replace(' ','N'))
df_test = pre_processing(dataset = df_test,
               to_drop = ['grau_instrucao',
                    'estado_onde_nasceu', 'estado_onde_reside',
                    'codigo_area_telefone_residencial',
                    'possui_telefone_celular',
                    'qtde_contas_bancarias_especiais', 'estado_onde_trabalha',
                    'codigo_area_telefone_trabalho', 'meses_no_trabalho',
                    'profissao_companheiro',
                    'grau_instrucao_companheiro', 'local_onde_reside',
                    'local_onde_trabalha'
                    ], 
               to_onehot = ['estado_civil', 'produto_solicitado', 'dia_vencimento',
                                      'forma_envio_solicitacao',
                                      'nacionalidade', 'sexo', 'tipo_residencia',
                                      'profissao', 'ocupacao', 'qtde_contas_bancarias'], 
               to_bin = ['tipo_endereco','possui_telefone_residencial',
                         'vinculo_formal_com_empresa', 
                         'possui_telefone_trabalho',
                         ], 
               )
#%%
#-------------------------------------------------------------------------------
# Matriz de Correlação
#-------------------------------------------------------------------------------
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,cmap=sns.color_palette("RdBu_r", 1000), vmin=-1,vmax=1, square=True)
plt.savefig('CorrelationMatrix.png')
#%%
#-------------------------------------------------------------------------------
# Selecionar os atributos que serão utilizados pelo classificador
#-------------------------------------------------------------------------------  
def select_attributes():
    atributos_selecionados = [
        'produto_solicitado_1',
        # 'produto_solicitado_2',
        'produto_solicitado_7',
        # 'dia_vencimento_1',
        # 'dia_vencimento_5',
        'dia_vencimento_10',
        # 'dia_vencimento_15',
        # 'dia_vencimento_20',
        'dia_vencimento_25',
        'forma_envio_solicitacao_correio',
        # 'forma_envio_solicitacao_internet',
        'forma_envio_solicitacao_presencial',
        # 'tipo_endereco',
        'sexo_F',
        # 'sexo_M',
        # 'sexo_N',
        'idade',
        'qtde_dependentes',
        # 'nacionalidade_0',
        # 'nacionalidade_1',
        # 'nacionalidade_2',
        'possui_telefone_residencial',
        # 'possui_email',
        # 'renda_mensal_regular',
        # 'renda_extra',
        # 'possui_cartao_visa',
        'possui_cartao_mastercard',
        # 'possui_cartao_diners',
        # 'possui_cartao_amex',
        # 'possui_outros_cartoes',
        # 'valor_patrimonio_pessoal',
        # 'possui_carro',                      # teste
        # 'vinculo_formal_com_empresa',
        'possui_telefone_trabalho',
        # 'estado_civil_0',
        'estado_civil_1',
        # 'estado_civil_2',
        # 'estado_civil_3',
        'estado_civil_4',
        # 'estado_civil_5',
        # 'estado_civil_6',
        # 'estado_civil_7',
        # 'tipo_residencia_0.0',
        'tipo_residencia_1.0',
        # 'tipo_residencia_2.0',
        # 'tipo_residencia_3.0',
        # 'tipo_residencia_4.0',
        # 'tipo_residencia_5.0',
        # 'ocupacao_0.0',
        'ocupacao_1.0',
        # 'ocupacao_2.0',
        # 'ocupacao_3.0',
        # 'ocupacao_4.0',
        # 'ocupacao_5.0',
        # 'profissao_0.0', 
        # 'profissao_1.0',
        # 'profissao_2.0',
        # 'profissao_3.0',
        # 'profissao_4.0',
        # 'profissao_5.0',
        # 'profissao_6.0',
        # 'profissao_7.0',
        # 'profissao_8.0',
        # 'profissao_9.0',
        # 'profissao_10.0',
        'profissao_11.0',
        # 'profissao_12.0',
        # 'profissao_13.0',
        # 'profissao_14.0',
        # 'profissao_15.0',
        # 'profissao_16.0',
        # 'profissao_17.0',
        'qtde_contas_bancarias_0',
        # 'qtde_contas_bancarias_1',
        # 'qtde_contas_bancarias_2',
        'meses_na_residencia'
    ]
    return atributos_selecionados
#-------------------------------------------------------------------------------
df_train    = df_train[select_attributes()  + ['inadimplente']]
df_test     = df_test[select_attributes()   + ['id_solicitante']]
#%%
#------------------------------------------------------------------------------
# Dividir Dataframe em X(Features) e y (target)
#------------------------------------------------------------------------------
X = df_train.loc[:,df_train.columns!='inadimplente'].values
y = df_train.loc[:,df_train.columns=='inadimplente'].values.ravel()
# X_2 e ids usado para avaliar conjunto_de_teste e criar csv de respostas
X_2 = df_test.loc[:,df_test.columns!='id_solicitante'].values   # Para conjunto de teste
ids = df_test.loc[:,df_test.columns=='id_solicitante'].values   # Salvar IDs
#%%
#------------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
#------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = seed)
#------------------------------------------------------------------------------
# Feature Scaling
#------------------------------------------------------------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_2 = sc.transform(X_2)

#%%
#-------------------------------------------------------------------------------
# Encontrar melhores hiper-parâmetros de um classificador (Hyperparameter Tuning):
#-------------------------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV
def find_hiperparameter(classifier, random_grid, x, y):
    # Use the random grid to search for best hyperparameters;
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    model_random = RandomizedSearchCV(estimator = classifier, 
                                      param_distributions = random_grid,
                                      n_iter = 100,
                                      cv = 3, 
                                      verbose = 2, 
                                      random_state = seed, 
                                      n_jobs = -1) 
    model_random.fit(x, y)     # Fit the random search model
    return model_random
#-------------------------------------------------------------------------------
# Selecionar hiper-parâmetros que serão buscados(RF)
#-------------------------------------------------------------------------------
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt'] # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)          # Maximum number of levels in tree
min_samples_split = [2, 5, 10]  # Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4]    # Minimum number of samples required at each leaf node
bootstrap = [True, False]     # Method of selecting samples for training each tree
#-------------------------------------------------------------------------------
# Create the random grid
#-------------------------------------------------------------------------------
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap
               }
# print(random_grid)
#%%
#-------------------------------------------------------------------------------
# Executando busca pelos melhores hiper-parâmetros (ETAPA OPCIONAL)
#-------------------------------------------------------------------------------
# Essa etapa pode demorar muito, de acordo com o grid criado, dependendo de quantas 
# combinações serão testadas.
# Caso queira, pule para etapa seguinte, de treinamento, usando parâmetros quaisquer.
# Obs.: O resultado das buscas para GBC e RF se encontram comentados abaixo, 
# podendo ser utilizadas na próxima etapa.
hyper_search = 'RF'
if (hyper_search == 'GBC'):
    # Tempo que levou para executar a busca: 5h13min
    model_hyper = find_hiperparameter(GradientBoostingClassifier(random_state = seed),
                        random_grid,
                        X_train,
                        y_train)
    print(model_hyper.best_params_)
    # Melhores parâmetros encontrados pela busca para o GBC:
    #   n_estimators = 200,
    #   min_samples_split = 5,
    #   min_samples_leaf = 1,
    #   max_features = 'sqrt',
    #   max_depth = 10,
    # Obs.: Essa primeira busca tinha random_state 42 na função 
    # RandomizedSearchCV. Posteriormente foi configurado um seed padrão, 
    # portanto uma nova busca pode gerar um resultado diferente.
if (hyper_search == 'RF'):
    # Tempo que levou para executar a busca: 36min
    model_hyper = find_hiperparameter(RandomForestClassifier(random_state = seed), 
                                random_grid,
                                X_train, 
                                y_train)
    print(model_hyper.best_params_)
    # Melhores parâmetros encontrados pela busca para o RandomForest:
    #   n_estimators = 1600,
    #   min_samples_split = 2,
    #   min_samples_leaf = 4,
    #   max_features = 'sqrt',
    #   max_depth = 10,
    #   bootstrap = True
    # Obs.: Essa primeira busca tinha random_state 42 na função 
    # RandomizedSearchCV. Posteriormente foi configurado um seed padrão, 
    # portanto uma nova busca pode gerar um resultado diferente.
#%%
#-------------------------------------------------------------------------------
# Treinar um modelo:
#-------------------------------------------------------------------------------
def train_model(classifier):
    if(classifier == 'LR'):
        model = LogisticRegression(random_state = seed)
        model.fit(X_train, y_train)
        return model
    if(classifier == 'KNN'):
        print ( "\n  K TREINO  TESTE")
        print ( " -- ------ ------")
        for k in range(1,130,2):
            model = KNeighborsClassifier(
                n_neighbors = k,
                weights     = 'uniform',
                metric = 'minkowski',
                p           = 2
                )
            model = model.fit(X_train,y_train)
            y_resposta_treino = model.predict(X_train)
            y_resposta_teste  = model.predict(X_test)  
            acuracia_treino = sum(y_resposta_treino==y_train)/len(y_train)
            acuracia_teste  = sum(y_resposta_teste ==y_test) /len(y_test)   
            print(
                "%3d"%k,
                "%6.1f" % (100*acuracia_treino),
                "%6.1f" % (100*acuracia_teste)
                )
        return model
    if(classifier == 'SV'):
        model = SVC(kernel = 'linear', random_state = seed) # kernel = 'rbf'
        model.fit(X_train, y_train)
        return model
    if(classifier == 'NB'):
        model = GaussianNB()
        model.fit(X_train, y_train)
        return model
    if(classifier == 'DT'):
        model = DecisionTreeClassifier(criterion = 'entropy', random_state = seed)
        model.fit(X_train, y_train)
        return model
    if(classifier == 'RF'):
        # Hiper-parâmetros selecionados após a busca:
        model  = RandomForestClassifier(n_estimators = 1600,
                              min_samples_split = 2,
                              min_samples_leaf = 4,
                              max_features = 'sqrt',
                              max_depth = 10,
                              bootstrap = True,
                              random_state = seed)
        model.fit(X_train, y_train)
        print(model.feature_importances_)
        return model
    if(classifier == 'RG'):
        model = RidgeClassifier(alpha = 1, class_weight = 'balanced', solver = 'auto')
        model.fit(X_train,y_train)
        return model
    if(classifier == 'GBC'):
        # Hiper-parâmetros selecionados após a busca:
        model = GradientBoostingClassifier(random_state = seed,
                                           n_estimators = 200,
                                           min_samples_split = 5,
                                           min_samples_leaf = 1,
                                           max_features = 'sqrt',
                                           max_depth = 10,)
        rfe = RFE(model)
        rfe = rfe.fit(X_train, y_train)
        return rfe
    if(classifier == 'MLP'):
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        cvscores = []
        for treino, teste in kfold.split(X_train, y_train):
          model = tf.keras.models.Sequential()
          model.add(tf.keras.layers.Dense(units=20, activation='relu'))
          model.add(tf.keras.layers.Dense(units=10, activation='relu'))
          model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
          model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
          model.fit(X_train, y_train, batch_size = 32, epochs = 100, verbose=0)
          scores = model.evaluate(X_test, y_test, verbose=0)
          print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
          cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        model.summary()   
        return model

model = train_model(classifier = 'RF')
#%%
#-------------------------------------------------------------------------------
# Avaliar modelo:
#-------------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
#%%
#-------------------------------------------------------------------------------
# Matriz de confusão:
#-------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(y_test, y_pred, classes=[1, 0], title='Confusion matrix')
#%%
#-------------------------------------------------------------------------------
# Curva ROC:
#-------------------------------------------------------------------------------
import scikitplot as skplt
y_true = y_test
y_probas = np.concatenate( (1-y_pred.reshape(5000,1), y_pred.reshape(5000,1)), axis = 1)
skplt.metrics.plot_roc(y_true, y_probas)
plt.show()
#%%
#-------------------------------------------------------------------------------
# Gerando arquivo de resposta para submissão:
#-------------------------------------------------------------------------------
y_pred_test = model.predict(X_2)
y_pred_test = (y_pred_test > 0.5)
# y_pred_test[:,None]
dataframe = pd.DataFrame(np.concatenate((ids,y_pred_test[:,None]), axis = 1), 
                         columns=['id_solicitante','inadimplente']) 
dataframe.to_csv('arquivoRF_f.csv', sep=',',decimal = '.', index=False)