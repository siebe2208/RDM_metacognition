library(tidyverse)
library(ggplot2)
library(rstudioapi)
library(cmdstanr)
library(posterior)
library(ordbetareg)
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

get_conf <- function(x, ACC, theta, alpha) {
  ifelse(ACC == 1 & x > alpha,  theta,
    ifelse(ACC == 1 & x < alpha, 1 - theta,
      ifelse(ACC == 0 & x > alpha, 1 - theta,
        ifelse(ACC == 0 & x < alpha, theta,0))))
}

## Data Cleaning
data = read.csv(here("Analysis", "RDM_reportz_sub0_2.csv"))
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

## Data extraction
N = nrow(data_main)
binom_y = data_main$cor
RT = data_main$RTdec
conf = data_main$SR_conf
minRT = min(data_main$RTdec)
x = data_main$coherence
ACC = data_main$cor

## Model fit 

data_list = list(N = N, RT = RT, Conf = conf, X = x, minRT = minRT, ACC = ACC, binom_y = binom_y)

model = cmdstan_model(here("Analysis", "single_sub_nocopula.stan"))

#fit = model$sample(data = data_list, parallel_chains = 4)
#fit$save_object(here("models", "fit_pilot_siebe.rds"))
fit <- readRDS(here("Analysis", "fit_pilot_siebe.rds"))

## Fit extraction Type 1
x = seq(-1,1, length.out = 200)
draw_id = sample(1:4000,100)

draws = as_draws_df(fit$draws(c("alpha","beta","lapse", "rt_int", "rt_slope", "rt_ndt", "rt_prec", "meta_un", "meta_bias", "conf_prec"))) %>% 
  select(-contains(".")) %>% mutate(conf_slope = exp(beta+meta_un))

posterior = draws %>%
  mutate(beta = exp(beta)) %>% mutate(draw = 1:n(),
  x = list(x)) %>% unnest(x)

draws_psych = posterior %>% filter(draw%in% draw_id) %>% 
  mutate(y_pred = psycho_ACC(x, alpha, beta, lapse))

# median parameters psychometric function 
med_alpha = median(posterior$alpha)
med_beta = median(posterior$beta)
med_lapse = median(posterior$lapse)
y_pred_med = psycho_ACC(x, med_alpha, med_beta, med_lapse)

# residuals Type 1
data_main = data_main %>% mutate(model_pred = psycho_ACC(coherence, med_alpha, med_beta, med_lapse))%>%
  mutate(residual = up-model_pred) 

## Fit extraction RT
cutpoints = as_draws_df(fit$draws(c("c0", "c11"))) %>% summarise(
  c0 = mean(c0),
  c11 = mean(c11)
)
  
posterior_1 = posterior %>% mutate(theta = psycho_ACC(x, alpha, beta, lapse), entropy = entropy(theta)) %>%
  mutate(rt_mu = rt_int + rt_slope*entropy) %>% mutate(rt_mu_ndt = rt_mu - rt_ndt) %>% rowwise() %>%
  mutate(rt_muEXP = exp(rt_mu)+rt_ndt) %>% 
  mutate(RT_sim = rlnorm(1, meanlog = rt_mu, sdlog = rt_prec) +rt_ndt) 


summary = posterior_1 %>% select("x", "draw", "rt_mu", "rt_prec", 'rt_ndt', "rt_mu_ndt", "RT_sim", "rt_muEXP") %>% arrange(x) %>% group_by(x) %>%
  summarise(
    med_rt = median(rt_muEXP, na.rm = TRUE),
    low = quantile(rt_muEXP, probs = 0.05),
    high = quantile(rt_muEXP, probs = 0.95),
    sd_low = quantile(RT_sim, probs = 0.05),
    sd_high = quantile(RT_sim, probs = 0.95)
  )

## Residuals RT type 1
data_main = data_main %>% rowwise() %>%
  mutate(RT_pred = list(draws$rt_int + draws$rt_slope * entropy(model_pred))) %>% 
  mutate(RT_pred = list(exp(RT_pred) + draws$rt_ndt)) %>% 
  mutate(RT_pred = mean(RT_pred)) %>%
  mutate(residuals_RT = RTdec -RT_pred)

## Fit extractions for confidence 
cutpoints = as_draws_df(fit$draws(c("c0", "c11"))) %>% summarise(
  c0 = mean(c0),
  c11 = mean(c11)
)
posterior_conf = posterior %>% select(-contains("rt")) %>%
  mutate(meta_un = exp(meta_un), 
         conf_slope = beta*meta_un,
         theta = psycho_ACC(x, alpha, beta, lapse),
         conf_theta = psycho_ACC(x, alpha, conf_slope, lapse),
         cor = rbinom(n(),1, p = theta)) %>%
           mutate(prob_cor_conf = get_conf(x, cor, conf_theta, alpha), conf_mu = inv.logit(logit(prob_cor_conf)+meta_bias)) %>%
  mutate(conf_sim = rordbeta(n = n(), mu = conf_mu, phi = conf_prec, cutpoints = c(cutpoints$c0, exp(cutpoints$c0) + cutpoints$c11)))

summary_conf = posterior_conf %>% arrange(x) %>% group_by(x) %>%
    summarise(
      med_conf_cor = median(conf_mu[cor == 1]),
      low_cor = quantile(conf_mu[cor==1], probs = 0.05),
      high_cor = quantile(conf_mu[cor==1], probs = 0.95),
      med_conf_incor = median(conf_mu[cor == 0]),
      low_incor = quantile(conf_mu[cor==0], probs = 0.05),
      high_incor = quantile(conf_mu[cor==0], probs = 0.95),
      sd_low_cor = quantile(conf_sim[cor==1], probs = 0.05),
      sd_high_cor = quantile(conf_sim[cor==1], probs = 0.95),
      sd_low_incor = quantile(conf_sim[cor==0], probs = 0.05),
      sd_high_incor = quantile(conf_sim[cor==0], probs = 0.95),
    )

## Residuals confidence 
data_main = data_main %>% rowwise() %>%
  mutate(conf_pred = list(psycho_ACC(coherence, draws$alpha, draws$conf_slope, draws$lapse))) %>% 
  mutate(conf_pred = list(get_conf(coherence, cor, conf_pred, draws$alpha))) %>%
  mutate(conf_pred = list(inv.logit(logit(conf_pred)+draws$meta_bias))) %>%
  mutate(conf_pred = mean(conf_pred)) %>%
  mutate(residuals_conf = SR_conf -conf_pred)
  
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

type_1_res =  ggplot()+
              geom_point(data = data_main, aes(x=coherence,y= residual), color = "grey", size = 4, shape = 16)+
              geom_hline(yintercept = 0, linetype = "longdash", color = "black", size = 1.2)+
              labs(x = "Coherence", y = "Obs - Pred", title = "residuals Type 1") +
              theme_minimal()+
              theme(plot.title = element_text(hjust = 0.5, size = 16),
                    axis.title = element_text(size = 16),                 
                    axis.text = element_text(size = 14), 
                    axis.line = element_line(size = 1.5, color = "black"))
type_1_p + type_1_res

## Plots reaction times
RT_p = ggplot()+
        geom_point(data = data_main, aes(x=coherence, y = RTdec), 
                  color = "blue", size =2.5, alpha =0.5)+
        geom_ribbon(data= summary, aes(x=x, ymin =sd_low, ymax = sd_high), fill ="grey", alpha = 0.3)+
        geom_ribbon(data = summary, aes(x = x, ymin = low, ymax = high), fill = "grey", alpha= 0.5)+
        geom_line(data = summary,aes(x = x, y = med_rt), color = "black", size = 1.2, alpha = 1)+
        labs(x = "Coherence", y = "Reaction times", title = "RT Type 1")+
        theme_minimal()+
        theme(plot.title = element_text(hjust = 0.5, size = 16),
              axis.title = element_text(size = 16),                 
              axis.text = element_text(size = 14), 
              axis.line = element_line(size = 1.5, color = "black"))

RT_res = ggplot() + 
          geom_point(data = data_main, aes(x=coherence,y= residuals_RT), color = "grey", size = 4, shape = 16)+
          geom_hline(yintercept = 0, linetype = "longdash", color = "black", linewidth = 1.2)+
          labs(x = "Coherence", y = "Obs - Pred", title = "residuals RT Type 1") +
          theme_minimal()+
          theme(plot.title = element_text(hjust = 0.5, size = 16),
                axis.title = element_text(size = 16),                 
                axis.text = element_text(size = 14), 
                axis.line = element_line(size = 1.5, color = "black"))
RT_p + RT_res

##Plots confidence
conf_p =  ggplot() +
            geom_point(data = data_main, aes(x = coherence, y = SR_conf, color = factor(cor, levels = c(0, 1), labels = c("Incorrect", "Correct"))), size =2.5, alpha =1, shape = 16) +
            scale_color_manual(values = c("Incorrect" = "salmon", "Correct" = "lightgreen")) +
            geom_smooth(data = summary_conf, aes(x=x, y=med_conf_cor), method = "loess", span =0.3, color = "darkgreen", size =1.2, linewidth = 1.2)+
            geom_ribbon(data= summary_conf, aes(x=x, ymin = low_cor, ymax = high_cor), fill = "darkgreen", alpha= 0.5)+
            geom_ribbon(data= summary_conf, aes(x=x, ymin = sd_low_cor, ymax = sd_high_cor), fill = "darkgreen", alpha= 0.1)+
            geom_smooth(data = summary_conf, aes(x=x, y=med_conf_incor), method = "loess", span = 0.3, color = "darkred", size =1.2, linewidth = 1.2)+
            geom_ribbon(data= summary_conf, aes(x=x, ymin = low_incor, ymax = high_incor), fill = "darkred", alpha= 0.5)+
            geom_ribbon(data= summary_conf, aes(x=x, ymin = sd_low_incor, ymax = sd_high_incor), fill = "darkred", alpha= 0.1)+
            theme_minimal() +
            labs(color = "Response", x = "Coherence", y = "Confidence", title = "Metacognition")+
            theme(
              plot.title = element_text(hjust = 0.5, size = 16),
              axis.title = element_text(size = 14),                 
              axis.text = element_text(size = 14), 
              axis.line = element_line(size = 1.5, color = "black"),
              legend.position= c(0.9, 0.5),
              legend.title = element_text(size = 16),  
              legend.text  = element_text(size = 14)   
            )

conf_res = ggplot() + 
            geom_point(data = data_main, aes(x=coherence,y= residuals_conf, color = factor(cor, levels = c(0, 1), labels = c("Incorrect", "Correct"))), size = 4, shape = 16)+
            scale_color_manual(values = c("Incorrect" = "salmon", "Correct" = "lightgreen"))+
            geom_hline(yintercept = 0, linetype = "longdash", color = "black", linewidth = 1.2)+
            labs(x = "Coherence", y = "Obs - Pred", title = "residuals Metacognition") +
            theme_minimal()+
            theme(plot.title = element_text(hjust = 0.5, size = 16),
                  legend.position = "none",
                  axis.title = element_text(size = 16),                 
                  axis.text = element_text(size = 14), 
                  axis.line = element_line(size = 1.5, color = "black"))

conf_p + conf_res


  