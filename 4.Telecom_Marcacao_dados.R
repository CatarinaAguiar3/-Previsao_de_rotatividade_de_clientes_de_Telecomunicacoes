# Carregar os pacotes

library(dplyr)
library(data.table)
library(tidyr)
library(ggplot2)
library(tidyverse)
library(psych) # função describe
library(ggcorrplot) # grafico de correlação
library(pastecs)
library(dummy) #Criar dummy: linhas passam a ser colunas binárias 
library(randomForest) #Usar no algoritmo Random Forest
library(boot)#Fazer CrossValidation
library(fastDummies) 
library(stats) #Métrica KS
#install.packages("ROCR")
library(ROCR)#para fazer a curva ROC
library(pROC)




#install.packages("caret")
#install.packages("randomForest")
library(lattice) #necessária para usar o pacote caret
library(caret) #Fazer avaliação dos modelos, padronizar os dados etc
library(randomForest) 


# Carregar a base de dados ------------------------------------------------

setwd("C:/0.Projetos/2.Telecom_customer_(Churn)/Scripts")
df_final1 <- read.csv("df_final_modelagem.csv")


# Código utilizado no inicio da modelagem --------------------------------


## Quebrar o dataset em teste, treino e validação --------------------------

## Definindo a semente para reproduzibilidade
set.seed(123)

## Índices para amostra de treino (70% dos dados)
indice_treino <- sample(1:nrow(df_final1), 0.7 * nrow(df_final1), replace=FALSE)

## Índices para amostra de teste (15% dos dados)
indice_teste <- sample(setdiff(1:nrow(df_final1), indice_treino), 0.15* nrow(df_final1), replace=FALSE) 

## Índices para amostra de validação (15% dos dados restantes)
indice_validacao <- setdiff(1:nrow(df_final1), c(indice_treino, indice_teste))


#Conjunto de dados dividos
dados_treino <- df_final1[indice_treino, ]
dados_teste <- df_final1[indice_teste, ]
dados_validacao <- df_final1[indice_validacao, ]

#OBS: Índices para amostra de treino (indices_treino): São os índices das linhas 
#do seu conjunto de dados original que foram selecionadas para compor o conjunto 
#de treino. Esses índices são utilizados para extrair as linhas correspondentes do
#conjunto de dados original. Essencialmente, indices_treino são os números que indicam 
#quais observações (linhas) do seu dataset original fazem parte do conjunto de treino


## Padronizar e Pré Processar os dados -------------------------------------

# Criar um objeto de pré-processamento com base nos dados de treinamento
preproc <- preProcess(dados_treino, method = c("range"))

# Aplicar o mesmo pré-processamento aos conjuntos de treinamento, teste e validação
dados_treino1<- predict(preproc, dados_treino)
dados_teste1 <- predict(preproc, dados_teste)
dados_validacao1<- predict(preproc, dados_validacao)

dados_treino1 <- data.frame(dados_treino1)
dados_teste1 <- data.frame(dados_teste1)
dados_validacao1 <- data.frame(dados_validacao1)




# Marcação dos dados ------------------------------------------------------
dados_treino2 <- cbind(indice_treino,dados_treino1 )
dados_validacao2 <- cbind(indice_validacao,dados_validacao1 )

#Tabela com dados marcados
DT:: datatable(dados_validação2, rownames = FALSE)
#, rownames = FALSE

# Modelo Final -----------------------------------------------------------
set.seed(123)
final_model <- randomForest(churn ~ eqpdays + months + change_mou + totrev +
                              mou_cvce_Mean + avgqty + rev_Mean + avgmou +
                              totcalls + adjqty+ adjmou + totmrc_Mean + totmou +
                              peak_vce_Mean + plcd_vce_Mean + complete_Mean + unan_vce_Mean +
                              avg6rev + drop_vce_Mean + ovrmou_Mean ,
                            data= dados_treino1, importance= T, cv=10, ntree = 500, 
                            mtry = 5, nodesize= 100, type = "classification", sampsize=20000 )
#Previsão
pred<- predict(final_model, newdata = dados_validacao2, type = "response")


# Métricas ---------------------------------------------------------------


  pred_rf4 <- predict(final_model, newdata = dados_validacao1, type = "response")
  prob_rf4 <- ifelse(pred_rf4 >= 0.444, 1, 0)
  prob_rf4 <- as.factor(prob_rf4)
  verdadeiro <- factor(dados_validacao1$churn)
  matrix_rf4 <- confusionMatrix(prob_rf4, verdadeiro, positive = "1")
  precision_rf4 <- matrix_rf4$byClass["Pos Pred Value"]
  recall_rf4 <- matrix_rf4$byClass["Sensitivity"]
  f1_score_rf4 <- 2 * (precision_rf4 * recall_rf4) / (precision_rf4 + recall_rf4)
  ks_rf4 <- ks.test(pred_rf4, as.numeric(verdadeiro))
  roc_rf4 <- roc(dados_validacao1$churn, pred_rf4)
  auc_valor <- auc(roc_rf4)
  cat("Precision:", precision_rf4, "\n")
  cat("Recall:", recall_rf4, "\n")
  cat("F1-Score:", f1_score_rf4, "\n")
  cat("Valor do KS:", ks_rf4$statistic, "\n")
  cat("AUC:", auc_valor, "\n")
  print(matrix_rf4$table)


f_rf4("final_model")




# Shap Value --------------------------------------------------------------
#https://stackoverflow.com/questions/65391767/shap-plots-for-random-forest-models
#https://www.r-bloggers.com/2022/06/visualize-shap-values-without-tears/
#https://rdrr.io/cran/kernelshap/man/kernelshap.html


#install.packages("kernelshap")
#install.packages("shapviz")
library(kernelshap)
library(shapviz)


library(keras)


install.packages("viridis")
library(viridis)

#Etapa 1: Criar uma amostra aleatoria
set.seed(123)  # Definindo uma semente para reproducibilidade
amostra <- dados_treino1[sample(nrow(dados_treino1), 200), ]

# Etapa 2: Calcular os valores SHAP do kernel 
# bg_X geralmente é um subconjunto pequeno (50-200 linhas) dos dados
#s <- kernelshap(final_model, dados_treino1[-1], bg_X = amostra)

setwd("C:/0.Projetos/2.Telecom_customer_(Churn)/Scripts")
#saveRDS(s, file = "resultado_kernelshap.rds")
s1 <- readRDS("resultado_kernelshap.rds")

s1$X
# Etapa 3: Transforme-os em um objeto shapviz
sv <- shapviz(s1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

#https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html


#Etapa 4: Criar os gráficos
#Gráfico 1
sv_importance(sv, kind = "bee", viridis_args = list("#FD4B4B" , "#F5E273"))
sv_importance(sv, kind = "bee", viridis_args =list(begin = 0.25, end = 0.85, option = "turbo"))
sv_importance(sv, kind = "bee", viridis_args =list(option = "turbo", color_bar_title= "turbo"))
sv_importance(sv, kind = "bee", viridis_args = list(option = "turbo"))
sv_importance(sv, kind = "bee", viridis_args =list(option = "civids"))
sv_importance(sv, kind = "bee", viridis_args =list(option = "viridis"))
sv_importance(sv, kind = "bee", viridis_args =list(option = "magma"))
sv_importance(sv, kind = "bee", viridis_args =list(option = "plasma"))
sv_importance(sv, kind = "bee", viridis_args =list(option = "inferno"))
sv_importance(sv, kind = "bee", viridis_args =list(option = "plasma"))
sv_importance(sv, kind = "bee", viridis_args =list(option = "mako"))


sv_importance(sv, kind = "beeswarm")



#Gráfico 2
sv_dependence(sv, v = "eqpdays", color_var = "auto")

#Gráfico 3 
sv_importance(sv)

#Gráfico 4
sv_importance(sv, kind = "bar")

sv_importance(sv, kind = "bar")

sv_20 = sv[, order(abs(sv), decreasing = TRUE)[1:20]]

# Plotar o gráfico
sv_importance(sv_20, kind = "bar")

#Gráfico 5
sv_importance(sv, kind = "both")

#Gráfico 6
sv_waterfall ( sv, row_id = 1 )
#f(x)= 0.713 -> é a probabilidade dada pelo modelo.
#Segundo o modelo este cliente faz churn.
sv_waterfall ( sv, row_id = 2 )
sv_waterfall ( sv, row_id = 3 )
sv_waterfall ( sv, row_id = 4 )
sv_waterfall ( sv, row_id = 5 )
sv_waterfall ( sv, row_id = 6 )
# * f(x)=previsão do modelo para este exemplo. E[f(X)] + soma(SHAP_VALUE)
# * E[F(X)] = O valor do valor base é o mesmo para todos os exemplos nos dados.
#    É igual ao churn médio no conjunto de dados
# * Os valores SHAP são todos os valores intermediários.
# * os valores SHAP nos dizem como os recursos contribuíram 
#   para a previsão quando comparados com a previsão média. Grandes valores positivos/negativos indicam que o recurso teve um impacto significativo na previsão do modelo 
#Gráfico 7
sv_force (sv, row_id = 1)



# Tabela para simulação da faixa de corte e do desconto -------------------

#Criar Tabela
tabela<- dados_validacao2 %>% 
  select(indice_validacao,churn)

tabela1 <- cbind(tabela, pred)




## Intervalo -------------------------------------------------------------
# Encontrar o valor mínimo e máximo da coluna
min <- min(tabela1$pred)
max <- max(tabela1$pred)

#Mínimo: 0.06966931 
#Máximo: 0.817634 


# Definir o intervalo
intervalo <- c(0.06966931, 0.817634)

# Especificar o número de intervalos desejados
num_intervalos <- 20

# Criar os intervalos
intervalos <- seq(intervalo[1], intervalo[2], length.out = num_intervalos + 1)

# Imprimir os intervalos
cat("Intervalos:", intervalos, "\n")

data.frame(intervalos)


##Tabela final -----------------------------------------------------------

#Tabela 2
tabela2 <- tabela1  %>% mutate(
  f1_pred = ifelse(0.06966931< pred & pred<= 0.1070675445,1, 0),
  f2_pred = ifelse(0.1070675445< pred & pred<= 0.1444657790, 1,0),
  f3_pred = ifelse(0.1444657790< pred & pred<= 0.1818640135,1, 0),
  f4_pred = ifelse(0.1818640135< pred & pred<= 0.2192622480 ,1, 0),
  f5_pred = ifelse(0.2192622480 < pred & pred <= 0.2566604825,1, 0),
  f6_pred = ifelse(0.2566604825 < pred & pred <= 0.2940587170,1, 0),
  f7_pred = ifelse(0.2940587170 < pred & pred <= 0.3314569515,1, 0),
  f8_pred = ifelse(0.3314569515 < pred & pred <= 0.3688551860,1, 0),
  f9_pred = ifelse(0.3688551860 < pred & pred <= 0.4062534205,1, 0),
  f10_pred = ifelse(0.4062534205 < pred & pred <= 0.4436516550,1, 0),
  f11_pred = ifelse(0.4436516550 < pred & pred <= 0.4810498895,1, 0),
  f12_pred = ifelse(0.4810498895 < pred & pred <= 0.5184481240,1, 0),
  f13_pred = ifelse(0.5184481240< pred & pred <= 0.5558463585,1, 0),
  f14_pred = ifelse(0.5558463585 < pred & pred <= 0.5932445930,1, 0),
  f15_pred = ifelse(0.5932445930 < pred & pred <= 0.6306428275,1, 0),
  f16_pred = ifelse(0.6306428275< pred & pred <= 0.6680410620,1, 0),
  f17_pred = ifelse(0.6680410620< pred & pred <= 0.7054392965 ,1, 0),
  f18_pred = ifelse(0.7054392965 < pred & pred <= 0.7428375310,1, 0),
  f19_pred = ifelse(0.7428375310< pred & pred <= 0.7802357655,1, 0),
  f20_pred = ifelse(0.7802357655< pred & pred <= 0.8176340000,1, 0)
)

tabela2 <- tabela2  %>% mutate(
  f1_verd = ifelse(churn==1 & 0.06966931< pred & pred<= 0.1070675445,1, 0),
  f2_verd = ifelse(churn==1 & 0.1070675445< pred & pred<= 0.1444657790, 1,0),
  f3_verd = ifelse(churn==1 & 0.1444657790< pred & pred<= 0.1818640135,1, 0),
  f4_verd = ifelse(churn==1 & 0.1818640135< pred & pred<= 0.2192622480, 1, 0),
  f5_verd = ifelse(churn==1 & 0.2192622480 < pred & pred <= 0.2566604825,1, 0),
  f6_verd = ifelse(churn==1 & 0.2566604825 < pred & pred <= 0.2940587170,1, 0),
  f7_verd = ifelse(churn==1 & 0.2940587170 < pred & pred <= 0.3314569515,1, 0),
  f8_verd = ifelse(churn==1 & 0.3314569515 < pred & pred <= 0.3688551860,1, 0),
  f9_verd = ifelse(churn==1 & 0.3688551860 < pred & pred <= 0.4062534205,1, 0),
  f10_verd = ifelse(churn==1 & 0.4062534205 < pred & pred <= 0.4436516550,1, 0),
  f11_verd = ifelse(churn==1 & 0.4436516550 < pred & pred <= 0.4810498895,1, 0),
  f12_verd = ifelse(churn==1 & 0.4810498895 < pred & pred <= 0.5184481240,1, 0),
  f13_verd = ifelse(churn==1 & 0.5184481240< pred & pred <= 0.5558463585,1, 0),
  f14_verd = ifelse(churn==1 & 0.5558463585 < pred & pred <= 0.5932445930,1, 0),
  f15_verd = ifelse(churn==1 & 0.5932445930 < pred & pred <= 0.6306428275,1, 0),
  f16_verd = ifelse(churn==1 & 0.6306428275< pred & pred <= 0.6680410620,1, 0),
  f17_verd = ifelse(churn==1 & 0.6680410620< pred & pred <= 0.7054392965,1, 0),
  f18_verd = ifelse(churn==1 & 0.7054392965 < pred & pred <= 0.7428375310,1, 0),
  f19_verd = ifelse(churn==1 & 0.7428375310< pred & pred <= 0.7802357655,1, 0),
  f20_verd = ifelse(churn==1 & 0.7802357655< pred & pred <= 0.8176340000,1, 0)
)

#Esta tabela será utilizada para criar a simulação no Excel
#Tabela Final
total_colunas <- colSums(select(tabela2, 4:43), na.rm = TRUE)
total_colunas1<- data.frame(total_colunas)
total_colunas1

#Verificar quantidade de churn
table(dados_validação2$churn)

setwd("C:/0.Projetos/2.Telecom_customer_(Churn)/Scripts")
write.csv2(total_colunas1, "tabela2_ponto_de_corte.csv")
