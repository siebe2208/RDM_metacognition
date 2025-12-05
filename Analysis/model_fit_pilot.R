library(tidyverse)
library(ggplot2)
library(rstudioapi)
library(cmdstanr)
library(posterior)
library(boot)
library(here)
library(patchwork)

## functions
psycho_ACC = function(x, alpha, beta, lapse){
  ACC = lapse + (1 - 2 * lapse) * inv.logit(beta * (x - alpha))
}

entropy = function(p){
  entropy = -p*log(p)-(1-p)*log(1-p)
  return(entropy)
}

## Data Cleaning
data = read.csv(choose.files())
data_main = data[data$Trialtype == "Main",]
data_main = data_main[data_main$scale == "conf", ]
data_main$SR_conf = as.numeric(data_main$SR_conf)
data_main$SR_conf = (data_main$SR_conf + 1)/2

data_main = data_main[!is.na(data_main$SR_conf), ]
sum(is.na(data_main$SR_conf))
data_main = data_main[!is.na(data_main$resp), ]
sum(is.na(data_main$resp))

data_main <- data_main %>%
  mutate(coherence = ifelse(cor_resp == "down", -(coherence), coherence))
data_main$up = ifelse(data_main$resp == "up", 1, 0)

#Data extraction
N = nrow(data_main)
binom_y = data_main$cor
RT = data_main$RTdec
conf = data_main$SR_conf
minRT = min(data_main$RTdec)
x = data_main$coherence
ACC = data_main$cor

## Model fit 

data_list = list(N = N, RT = RT, Conf = conf, X = x, minRT = minRT, ACC = ACC, binom_y = binom_y)

model = cmdstan_model(here("models", "single_sub_nocopula.stan"))

#fit = model$sample(data = data_list, parallel_chains = 4)
#fit$save_object(here("models", "fit_pilot_siebe.rds"))
fit <- readRDS("~/psychologie 1e Master/internship/models/fit_pilot_siebe.rds")

## Fit extraction Type 1
x = seq(-1,1, length.out = 200)
draw_id = sample(1:4000,100)

posterior = as_draws_df(fit$draws(c("alpha","beta","lapse"))) %>%
  mutate(beta = exp(beta)) %>%
  select(-contains(".")) %>% mutate(draw = 1:n(),
  x = list(x)) %>% unnest()

draws_psych = posterior %>% filter(draw%in% draw_id) %>% 
  mutate(y_pred = psycho_ACC(x, alpha, beta, lapse))

# median parameters psychometric function 
med_alpha = median(posterior$alpha)
med_beta = median(posterior$beta)
med_lapse = median(posterior$lapse)
y_pred_med = psycho_ACC(x, med_alpha, med_beta, med_lapse)

# residuals 
residuals = data_main %>% mutate(model_pred = psycho_ACC(coherence, med_alpha, med_beta, med_lapse))%>%
  mutate(residual = up-model_pred) %>% pull(residual)

## Fit extraction RT
draws_RT = as_draws_df(fit$draws(c("rt_int", "rt_slope", "rt_ndt"))) %>%  
  select(-contains(".")) %>% mutate(draw = 1:n(),
  x = list(x))%>% unnest()
med_rt_int = median(draws_RT$rt_int)
med_rt_slope = median(draws_RT$rt_slope)
med_rt_ndt = median(draws_RT$rt_ndt)
entropy_pred = entropy(y_pred_med)
RT_pred = exp(med_rt_int + med_rt_slope*entropy_pred) + med_rt_ndt

## Plots binary choice
type_1_p = ggplot() +
            geom_point(data = data_main, aes(x = coherence, y = up), color = "red", size = 2, shape = 16, alpha = 0.7) +
            geom_line(data = draws_psych, aes(x=x, y = y_pred, group = draw), col = "black", alpha = 0.05)+
            geom_line(aes(x=x, y=y_pred_med), color = "black", size = 0.7, alpha = 1)+
            labs(x = "Coherence", y = "P('up')", title = "Type 1") +
            theme_minimal()+
            theme(plot.title = element_text(hjust = 0.5, size = 16),
                  axis.title = element_text(size = 16),                 
                  axis.text = element_text(size = 14), 
                  axis.line = element_line(size = 1.5, color = "black"))

type_1_res = data.frame(x = data_main$coherence, y=residuals)%>% 
              ggplot()+
              geom_point(aes(x=x,y=y), color = "grey", size = 4, shape = 16)+
              geom_hline(yintercept = 0, linetype = "longdash", color = "black", size = 1.2)+
              labs(x = "Coherence", y = "Obs - Pred", title = "residuals Type 1") +
              theme_minimal()+
              theme(plot.title = element_text(hjust = 0.5, size = 16),
                    axis.title = element_text(size = 16),                 
                    axis.text = element_text(size = 14), 
                    axis.line = element_line(size = 1.5, color = "black"))
type_1_p + type_1_res

## Plots reaction times
ggplot()+
  geom_point(data = data_main, aes(x=coherence, y = RTdec), 
            color = "grey", size =2.5, alpha =0.5)+
  geom_line(aes(x = x, y = RT_pred), color = "black", size = 1, alpha = 1)+
  labs(x = "Coherence", y = "Reaction times", title = "RT Type 1")+
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 16),                 
        axis.text = element_text(size = 14), 
        axis.line = element_line(size = 1.5, color = "black"))
  